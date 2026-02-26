"""TSFM (Time Series Foundation Model) MCP Server.

Standalone implementation — no dependency on src/tsfmagent/.
The only external ML dependency is tsfm_public (IBM Granite TSFM):
  https://github.com/IBM-granite/granite-tsfm

Tools:
  get_ai_tasks          – list available AI task types (static, no deps)
  get_tsfm_models       – list available pre-trained model checkpoints (static)
  run_tsfm_forecasting  – zero-shot TTM inference on a dataset
  run_tsfm_finetuning   – few-shot finetuning of a TTM model
  run_tsad              – conformal anomaly detection on TSFM forecasts
  run_integrated_tsad   – end-to-end: forecasting + anomaly detection

Heavy ML dependencies (tsfm_public, transformers, torch) are imported lazily;
the server starts and exposes the static tools even when they are absent.

Required environment variables (path resolution):
  PATH_TO_MODELS_DIR    – directory containing TTM model checkpoint folders
  PATH_TO_DATASETS_DIR  – base directory for resolving relative dataset paths
  PATH_TO_OUTPUTS_DIR   – base directory for resolving output/save paths
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import tempfile
import time
import uuid
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .models import (
    _AI_TASKS,
    _TSFM_MODELS,
    AITaskEntry,
    AITasksResult,
    ErrorResult,
    FinetuningResult,
    ForecastingResult,
    TSADResult,
    TSFMModelEntry,
    TSFMModelsResult,
)

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")


# ── Path / I/O helpers ────────────────────────────────────────────────────────


def _get_model_checkpoint_path(model_checkpoint: str) -> str:
    if os.path.isabs(model_checkpoint):
        return model_checkpoint
    return os.path.join(os.environ.get("PATH_TO_MODELS_DIR", ""), model_checkpoint)


def _get_dataset_path(dataset: str) -> str:
    if os.path.isabs(dataset):
        return dataset
    return os.path.join(os.environ.get("PATH_TO_DATASETS_DIR", ""), dataset)


def _get_outputs_path(outputs: str) -> str:
    if os.path.isabs(outputs):
        return outputs
    return os.path.join(os.environ.get("PATH_TO_OUTPUTS_DIR", ""), outputs)


def _write_json_to_temp(json_data: str) -> str:
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}.json")
    with open(temp_file_path, "w") as f:
        f.write(json_data)
    return temp_file_path


def _make_json_compatible(obj):
    """Recursively convert an object to a JSON-serializable form."""
    if isinstance(obj, dict):
        return {str(k): _make_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_compatible(i) for i in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


# ── Dataset reading ───────────────────────────────────────────────────────────


def _read_ts_data(dataset_path: str, dataset_config_dictionary=None) -> pd.DataFrame:
    if dataset_config_dictionary is not None:
        timestamp_column = dataset_config_dictionary["column_specifiers"][
            "timestamp_column"
        ]
    else:
        timestamp_column = "Date"

    valid_extensions = {".csv", ".json", ".xlsx"}
    _, file_extension = os.path.splitext(dataset_path)
    if file_extension.lower() not in valid_extensions:
        raise ValueError(
            f"Invalid file type received: {dataset_path}. "
            f"Expected file types are: {', '.join(valid_extensions)}."
        )

    if ".csv" in dataset_path:
        if dataset_config_dictionary is not None:
            col_spec = dataset_config_dictionary["column_specifiers"]
            data_df = pd.read_csv(
                dataset_path, parse_dates=[col_spec["timestamp_column"]]
            )
        else:
            data_df = pd.read_csv(dataset_path)

    elif ".json" in dataset_path:
        try:
            with open(dataset_path) as _f:
                result_dict = json.load(_f)
            ts: dict = {}
            for input_data in result_dict:
                dt = datetime.fromisoformat(input_data["timestamp"])
                if (dt.minute % 15) != 0:
                    continue
                input_data[timestamp_column] = dt
                ts[dt] = input_data
            data_df = pd.DataFrame()
            for dt, v in ts.items():
                data_df = pd.concat([data_df, pd.DataFrame(v, index=[dt])])
        except Exception as ex:
            raise ValueError(
                f"input file {dataset_path} is not in the correct format"
            ) from ex

    elif ".xlsx" in dataset_path:
        if dataset_config_dictionary is not None:
            col_spec = dataset_config_dictionary["column_specifiers"]
            data_df = pd.read_excel(
                dataset_path, parse_dates=[col_spec["timestamp_column"]]
            )
        else:
            data_df = pd.read_excel(dataset_path)

    else:
        raise ValueError(
            f"file extension must be: .json, .csv, or .xlsx. file: {dataset_path}"
        )

    return data_df


# ── Data quality helpers ──────────────────────────────────────────────────────


def _threshold_condition_function(threshold, condition_type="<"):
    conditions = {
        "<": lambda x: x < threshold,
        "<=": lambda x: x <= threshold,
        ">": lambda x: x > threshold,
        ">=": lambda x: x >= threshold,
        "==": lambda x: x == threshold,
    }
    assert condition_type in conditions, (
        f"condition_type {condition_type!r} is not supported"
    )
    return conditions[condition_type]


def _df_nan_stats(df, perc_rows_less_than=None, perc_rows_more_than=None):
    if perc_rows_less_than is None:
        perc_rows_less_than = [10, 20, 50]
    if perc_rows_more_than is None:
        perc_rows_more_than = [50, 100]
    output: dict = {}
    nan_per_column = df.isna().mean() * 100
    output["%NaN_per_column"] = nan_per_column.to_dict()
    output["%rows_0_NaNs"] = (df.isna().sum(axis=1) == 0).mean() * 100
    if perc_rows_less_than and output["%rows_0_NaNs"] > 0:
        output["%rows_less_than"] = {
            f"{p}% NaNs": np.mean(df.isna().mean(axis=1) <= float(p / 100)) * 100
            for p in perc_rows_less_than
        }
    if perc_rows_more_than and output["%rows_0_NaNs"] > 0:
        output["%rows_more_than"] = {
            f"{p}% NaNs": np.mean(df.isna().mean(axis=1) > float(p / 100)) * 100
            for p in perc_rows_more_than
        }
    return output


def _df_percentage_samples_minutes_interval(
    df, date_col, lower_bound=14, upper_bound=16
):
    assert upper_bound >= lower_bound, "lower bound is larger than upper bound"
    df = df.sort_values(by=date_col)
    time_diffs = df[date_col].diff().dt.total_seconds() / 60.0
    interval_count = ((time_diffs >= lower_bound) & (time_diffs <= upper_bound)).sum()
    total_intervals = len(time_diffs) - 1
    return (interval_count / total_intervals) * 100 if total_intervals > 0 else 0


def _df_dt_stats(pd_dataset, date_col="Timestamp", intervals_dic=None):
    if intervals_dic is None:
        intervals_dic = {"14min_to_16min": (14, 16)}
    pd_dataset = pd_dataset.sort_values(by=date_col)
    earliest_date = pd_dataset[date_col].min()
    latest_date = pd_dataset[date_col].max()
    date_interval = latest_date - earliest_date
    time_intervals = pd_dataset[date_col].diff()
    time_intervals_dic = time_intervals.value_counts().to_dict()
    time_intervals_dic_json = {str(k): int(v) for k, v in time_intervals_dic.items()}
    perc_in_interval_dic = None
    if isinstance(intervals_dic, dict):
        perc_in_interval_dic = {
            key: _df_percentage_samples_minutes_interval(
                pd_dataset, date_col, lower_bound=bounds[0], upper_bound=bounds[1]
            )
            for key, bounds in intervals_dic.items()
        }
    data_specs: dict = {
        "initial_time": earliest_date.isoformat(),
        "final_time": latest_date.isoformat(),
        "interval": str(date_interval),
        "columns": pd_dataset.columns.values.tolist(),
        "number_samples": len(pd_dataset),
        "time_interval_between_samples": time_intervals_dic_json,
    }
    if perc_in_interval_dic is not None:
        data_specs["percentage_in_dt"] = perc_in_interval_dic
    return data_specs


def _df_single_columns_condition(df, condition_dic=None):
    if condition_dic is None:
        condition_dic = {}
    condition_count = {}
    for key, (column_name, condition) in condition_dic.items():
        if column_name in df.columns:
            mask = df[column_name].apply(condition)
            condition_count[key] = {
                "nsamples": int(np.sum(mask)),
                "percentile": 100 * np.sum(mask) / max(len(mask), 1),
            }
    return condition_count


def _efficient_nan_removal(pd_table, preference_tie="row"):
    def compute_removal_costs(df):
        row_non_nan = df.notna().sum(axis=1)
        col_non_nan = df.notna().sum(axis=0)
        row_costs = np.where(df.isna().sum(axis=1) == 0, np.inf, row_non_nan)
        col_costs = np.where(df.isna().sum(axis=0) == 0, np.inf, col_non_nan)
        return row_costs, col_costs

    def remove_lowest_cost(df, row_costs, col_costs, preference_tie="row"):
        min_row = np.min(row_costs)
        min_col = np.min(col_costs)
        if (min_row < min_col) or (min_row == min_col and preference_tie == "row"):
            row_to_remove = np.where(row_costs == min_row)[0]
            df = df.drop(df.index[row_to_remove])
            removed = f"Row {row_to_remove} removed with cost {min_row}"
        else:
            col_to_remove = np.where(col_costs == min_col)[0]
            df = df.drop(df.columns[col_to_remove], axis=1)
            removed = f"Column {col_to_remove} removed with cost {min_col}"
        return df, removed

    df_t = pd_table.copy()
    max_actions = len(df_t) + len(df_t.columns)
    t = 0
    actions = []
    while df_t.isna().any().any() and t < max_actions:
        row_costs, col_costs = compute_removal_costs(df_t)
        df_t, removed = remove_lowest_cost(df_t, row_costs, col_costs, preference_tie)
        actions.append(removed)
        t += 1
    cost_total = pd_table.notna().sum().sum() - df_t.notna().sum().sum()
    return {"df_filter": df_t, "actions": actions, "cost_total": cost_total}


def _remove_df_nans(df, p=50, dim="columns"):
    threshold = p / 100.0
    assert dim in ("columns", "rows")
    if dim == "columns":
        cols_to_drop = df.columns[df.isna().mean() > threshold]
        return df.drop(columns=cols_to_drop)
    rows_to_drop = df.index[df.isna().mean(axis=1) > threshold]
    return df.drop(index=rows_to_drop)


def _remove_df_rows_by_single_column_condition(df, column_name, condition):
    if column_name in df.columns:
        mask = df[column_name].apply(condition)
        return df[(1 - mask) == 1]
    return df


def _time_series_frequency_interval_segmentation(
    df, time_column, lower_bound=14, upper_bound=16
):
    df = df.sort_values(by=time_column).reset_index(drop=True)
    df["dt"] = df[time_column].diff().dt.total_seconds() / 60.0
    df["segment_id"] = 0
    segment_id = 0
    start_idx = 0
    for i in range(1, len(df)):
        if not (lower_bound <= df["dt"].iloc[i] <= upper_bound):
            df.loc[start_idx:i, "segment_id"] = segment_id
            segment_id += 1
            start_idx = i
    df.loc[start_idx:, "segment_id"] = segment_id
    return df.drop(columns="dt")


def _validate_time_series_segments(
    df_segment,
    segment_tag="segment_id",
    timestamp_tag="Timestamp",
    p_nan_rows=1,
    p_nan_columns=70,
    condition_off_dic=None,
    dt_bounds=None,
):
    if dt_bounds is None:
        dt_bounds = [14, 16]
    bad_quality_segments: dict = {}
    lower_bound, upper_bound = dt_bounds[0], dt_bounds[1]
    for seg_id in df_segment[segment_tag].unique():
        df_seg_i = df_segment.loc[df_segment[segment_tag] == seg_id]
        dic_nan = _df_nan_stats(
            df_seg_i, perc_rows_less_than=[p_nan_rows], perc_rows_more_than=[p_nan_rows]
        )
        if condition_off_dic is not None:
            df_cond = _df_single_columns_condition(
                df_seg_i, condition_dic=condition_off_dic
            )
        nan_cols = list(dic_nan["%NaN_per_column"].values())
        qc: dict = {
            "nan_per_column": np.max(np.array(nan_cols)) <= p_nan_columns,
            "nan_per_rows": list(dic_nan["%rows_more_than"].items())[0][1]
            <= p_nan_rows,
        }
        if condition_off_dic is not None:
            cond_vals = [df_cond[k]["nsamples"] for k in df_cond]
            qc["condition_off"] = np.max(np.array(cond_vals)) == 0
        perc = _df_percentage_samples_minutes_interval(
            df_seg_i,
            date_col=timestamp_tag,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        qc["sampling_dt_condition"] = perc == 100
        if not all(qc.values()):
            bad_quality_segments[seg_id] = qc
    return bad_quality_segments


def _time_series_segment_quality_summary(df, timestamp_column, segments_column):
    ts_cont_segments: dict = {}
    for ix_s in df[segments_column].unique():
        df_filter = df.loc[df[segments_column] == ix_s]
        ts_cont_segments[ix_s] = {
            "start": pd.to_datetime(df_filter[timestamp_column].values[0]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": pd.to_datetime(df_filter[timestamp_column].values[-1]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "samples": len(df_filter),
            "%nans": df_filter.isna().mean().mean() * 100,
        }
    return ts_cont_segments


_FILTERING_PARAMS_DEFAULT = {
    "nans": {"efficient_removal": {"preference_tie": "row"}},
    "dt": {"lower_bound": 14, "upper_bound": 16},
}


def _dq_timeseries_segmentation(
    pd_merge, filtering_params=None, timestamp_tag="Timestamp"
):
    if filtering_params is None:
        filtering_params = _FILTERING_PARAMS_DEFAULT
    df_cleaned = pd_merge.copy()
    p_nan_columns = 70
    p_nan_rows = 1

    if "nans" in filtering_params:
        if "efficient_removal" in filtering_params["nans"]:
            preference_tie = filtering_params["nans"]["efficient_removal"].get(
                "preference_tie", "row"
            )
            output_efficient = _efficient_nan_removal(
                df_cleaned, preference_tie=preference_tie
            )
            df_cleaned = output_efficient["df_filter"]
            p_nan_columns = 1
            p_nan_rows = 1
        if "p_nan_columns" in filtering_params["nans"]:
            p_nan_columns = filtering_params["nans"]["p_nan_columns"]
            df_cleaned = _remove_df_nans(df_cleaned, p=p_nan_columns, dim="columns")
        if "p_nan_rows" in filtering_params["nans"]:
            p_nan_rows = filtering_params["nans"]["p_nan_rows"]
            df_cleaned = _remove_df_nans(df_cleaned, p=p_nan_rows, dim="rows")

    df_cleaned[timestamp_tag] = pd.to_datetime(
        df_cleaned[timestamp_tag], errors="coerce"
    )
    df_cleaned = df_cleaned.dropna(subset=[timestamp_tag])

    condition_off_dic = None
    if "operation_condition" in filtering_params:
        operation_condition = filtering_params["operation_condition"]
        condition_off_dic = {}
        for op in operation_condition:
            col = operation_condition[op]["column"]
            if col in pd_merge.columns:
                condition_off_dic[op] = (
                    col,
                    _threshold_condition_function(
                        operation_condition[op]["threshold"],
                        condition_type=operation_condition[op]["condition_type"],
                    ),
                )
            else:
                logger.debug("Column %s not present in the cleaned dataset", col)
        if condition_off_dic:
            for key in condition_off_dic:
                df_cleaned = _remove_df_rows_by_single_column_condition(
                    df_cleaned, condition_off_dic[key][0], condition_off_dic[key][1]
                )

    lower_bound = filtering_params["dt"]["lower_bound"]
    upper_bound = filtering_params["dt"]["upper_bound"]
    df_segment = _time_series_frequency_interval_segmentation(
        df_cleaned, timestamp_tag, lower_bound=lower_bound, upper_bound=upper_bound
    )

    bad_quality_segments = _validate_time_series_segments(
        df_segment,
        segment_tag="segment_id",
        timestamp_tag=timestamp_tag,
        p_nan_rows=p_nan_rows,
        p_nan_columns=p_nan_columns,
        condition_off_dic=condition_off_dic,
        dt_bounds=[lower_bound, upper_bound],
    )
    if bad_quality_segments:
        for seg_id in bad_quality_segments:
            df_segment = df_segment[df_segment["segment_id"] != seg_id]
    return df_segment


# ── Forecasting metrics ───────────────────────────────────────────────────────


def _RMSE(y_true, y_pred, axis=None):
    values = (
        np.mean((y_true - y_pred) ** 2)
        if axis is None
        else np.mean((y_true - y_pred) ** 2, axis=axis)
    )
    return np.sqrt(values)


def _MAE(y_true, y_pred, axis=None):
    if axis is None:
        return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred), axis=axis)


def _MAPE(y_true, y_pred, axis=None):
    non_zero_mask = np.array(y_true != 0).astype("int")
    y_true_denom = np.array(y_true) * non_zero_mask + (1 - non_zero_mask) * 1e-15
    values = np.abs((y_true - y_pred)) / np.abs(y_true_denom)
    if axis is None:
        if np.sum(non_zero_mask) > 0:
            return np.sum(values * non_zero_mask) / np.sum(non_zero_mask) * 100
        return None
    if np.sum(non_zero_mask) > 0:
        numerator = np.sum(values * non_zero_mask, axis=axis)
        denominator = np.sum(non_zero_mask, axis=axis)
        output = 100 * numerator
        output[denominator == 0] = None
        output[denominator != 0] /= denominator[denominator != 0]
        return output
    return None


def _SMAPE(y_true, y_pred, axis=None):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    non_zero_mask = np.array(denominator != 0).astype("int")
    denominator = np.array(denominator) * non_zero_mask + (1 - non_zero_mask) * 1e-15
    values = np.abs(y_true - y_pred) / denominator
    if axis is None:
        return np.mean(values) * 100
    return np.mean(values, axis=axis) * 100


def _WAPE(y_true, y_pred, axis=None):
    numerator = np.abs((y_true - y_pred))
    if axis is None:
        denominator = np.sum(np.abs(y_true))
        if denominator == 0:
            return None
        return np.sum(numerator) / denominator
    denominator = np.sum(np.abs(y_true), axis=axis)
    values = np.sum(numerator, axis=axis)
    values[denominator == 0] = None
    values[denominator != 0] /= denominator[denominator != 0]
    return values


def _Bias(y_true, y_pred, axis=None):
    delta = y_pred - y_true
    if axis is None:
        return np.mean(delta)
    return np.mean(delta, axis=axis)


def _NRMSE(y_true, y_pred, axis=None, norm="mean"):
    values = _RMSE(y_true, y_pred, axis=axis)
    den = (np.max(y_true) - np.min(y_true)) if norm == "minmax" else np.mean(y_true)
    return np.sqrt(values) / np.abs(den)


def _cosine_similarity_matrix(A, B, axis=1):
    dot_product = np.sum(A * B, axis=axis)
    norm_A = np.linalg.norm(A, axis=axis)
    norm_B = np.linalg.norm(B, axis=axis)
    return dot_product / (norm_A * norm_B)


def _loss_helper(outputs, targets, fn, axis=1):
    import torch

    outputs = outputs.astype(np.float64)
    targets = targets.squeeze()
    outputs = outputs.squeeze()
    if len(targets.shape) == 0 or targets.shape[0] == 0:
        return np.array(0.0)
    if len(targets.shape) == 1:
        if targets.shape[0] < 4:
            return np.array([0.0])
        return (
            fn(
                torch.from_numpy(targets.reshape(1, -1)),
                torch.from_numpy(outputs.reshape(1, -1)),
            )
            .cpu()
            .detach()
            .item()
        )
    B, T = targets.shape
    if axis == 1:
        if T < 4:
            return np.array([0.0])
        return (
            fn(torch.from_numpy(targets), torch.from_numpy(outputs))
            .cpu()
            .detach()
            .numpy()
        )
    return np.array(0.0)


def _amp_loss(outputs, targets):
    import torch

    B, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_f = torch.fft.fft(outputs, fft_size, dim=-1)
    tgt_f = torch.fft.fft(targets, fft_size, dim=-1)
    out_norm = torch.linalg.vector_norm(outputs, dim=-1)
    tgt_norm = torch.linalg.vector_norm(targets, dim=-1)
    auto_corr = torch.fft.ifft(tgt_f * tgt_f.conj(), dim=-1).real
    auto_corr = torch.cat([auto_corr[..., -(T - 1) :], auto_corr[..., :T]], dim=-1)
    norm = torch.where(tgt_norm == 0, 1e-9, tgt_norm * tgt_norm)
    nac_tgt = auto_corr / norm.unsqueeze(1)
    cross_corr = torch.fft.ifft(tgt_f * out_f.conj(), dim=-1).real
    cross_corr = torch.cat([cross_corr[..., -(T - 1) :], cross_corr[..., :T]], dim=-1)
    norm2 = torch.where(tgt_norm * out_norm == 0, 1e-9, tgt_norm * out_norm)
    nac_out = cross_corr / (tgt_norm * out_norm).unsqueeze(1)
    return torch.mean(torch.abs(nac_tgt - nac_out), dim=-1)


def _ashift_loss(outputs, targets):
    import torch

    B, T = outputs.shape
    return T * torch.mean(
        torch.abs(1 / T - torch.softmax(outputs - targets, dim=-1)), dim=-1
    )


def _phase_loss(outputs, targets):
    import torch

    B, T = outputs.shape
    out_f = torch.fft.fft(outputs, dim=-1)
    tgt_f = torch.fft.fft(targets, dim=-1)
    tgt_f_sq = tgt_f.real**2 + tgt_f.imag**2
    mask = (tgt_f_sq > T).float()
    topk_indices = tgt_f_sq.topk(k=int(T**0.5), dim=-1).indices
    mask = mask.scatter_(-1, topk_indices, 1.0)
    mask[..., 0] = 1.0
    mask = torch.where(mask > 0, 1.0, 0.0)
    mask = mask.bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask, dim=-1).unsqueeze(1)
    zero_error = torch.abs(out_f) * not_mask
    zero_error = torch.where(
        torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error
    )
    mask_f = mask.float()
    mask_f /= torch.mean(mask_f, dim=-1).unsqueeze(1)
    ae = torch.abs(out_f - tgt_f) * mask_f
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    return (torch.mean(zero_error, dim=-1) + torch.mean(ae, dim=-1)) / (T**0.5)


def _tildeq_loss(target, output):
    amp = _amp_loss(target, output)
    shift = _ashift_loss(target, output)
    phase = _phase_loss(target, output)
    return 0.5 * phase + 0.5 * shift + 0.01 * amp


def _TILDEQ(outputs, targets, axis=1):
    return _loss_helper(outputs, targets, _tildeq_loss, axis=axis)


def _derivatives(inp, device="cpu"):
    import torch

    batch_size, lens = inp.shape[0:2]
    input2 = inp[:, 2:lens].to(device)
    input1 = inp[:, 0 : lens - 2].to(device)
    return input2 - input1


def _w_mse(targets, outputs, device="cpu"):
    import torch

    batch_size, lens = targets.shape[0:2]
    t1 = targets[:, 1:lens].to(device)
    t2 = targets[:, 0 : lens - 1].to(device)
    o1 = outputs[:, 1:lens].to(device)
    o2 = outputs[:, 0 : lens - 1].to(device)
    sigma = torch.tanh((t1 - t2) * (o1 - o2))
    nt = targets[:, 1:lens].to(device)
    no = outputs[:, 1:lens].to(device)
    return torch.abs(no - nt) * (1.0 - sigma)


def _trend_loss(targets, outputs, alpha=0.5, device="cpu"):
    import torch

    sq_error = _w_mse(targets, outputs, device)
    error1 = torch.mean(sq_error, dim=-1)
    x1 = _derivatives(targets, device)
    x2 = _derivatives(outputs, device)
    _xt1 = x1.squeeze()
    _xt2 = x2.squeeze()
    if len(_xt1.shape) == 1:
        _xt1 = torch.reshape(_xt1, (1, _xt1.shape[0]))
    _xt1 = _xt1.T
    _xt2 = _xt2.T
    xc1 = (_xt1 - _xt1.mean(dim=0)).T
    xc2 = (_xt2 - _xt2.mean(dim=0)).T
    p_corr = torch.nn.functional.cosine_similarity(xc1, xc2, dim=-1)
    w_corr = 1 - p_corr
    dd = torch.norm(targets - outputs, dim=-1)
    return error1 + alpha * w_corr * dd


def _TREND(targets, outputs, axis=1):
    return _loss_helper(outputs, targets, _trend_loss, axis=axis)


_METRICS_FORECAST = {
    "RMSE": _RMSE,
    "MAE": _MAE,
    "MAPE": _MAPE,
    "SMAPE": _SMAPE,
    "WAPE": _WAPE,
    "Bias": _Bias,
    "NRMSE": _NRMSE,
    "TREND": _TREND,
    "TILDEQ": _TILDEQ,
    "COSSIM": _cosine_similarity_matrix,
}


# ── Frequency token mappings ──────────────────────────────────────────────────

_freq_token_mapping = {
    "oov": 0,
    "minutely": 1,
    "2_minutes": 2,
    "5_minutes": 3,
    "10_minutes": 4,
    "15_minutes": 5,
    "half_hourly": 6,
    "hourly": 7,
}
_freq_token_to_minutes = {
    "oov": None,
    "minutely": 1,
    "2_minutes": 2,
    "5_minutes": 5,
    "10_minutes": 10,
    "15_minutes": 15,
    "half_hourly": 30,
    "hourly": 60,
}
_TSFREQUENCY_TOLERANCE = 0.2


# ── TSFM data quality filter ──────────────────────────────────────────────────


def _tsfm_data_quality_filter(
    df_dataframe, dataset_config_dictionary, model_config, task="inference"
):
    timestamp_col = dataset_config_dictionary["column_specifiers"]["timestamp_column"]
    data_col = [timestamp_col]
    for columns_group in dataset_config_dictionary["column_specifiers"]:
        if "_columns" in columns_group:
            data_col.extend(
                dataset_config_dictionary["column_specifiers"][columns_group]
            )
    if "operation_on_column" in dataset_config_dictionary:
        data_col.extend(dataset_config_dictionary["operation_on_column"])

    df = df_dataframe[data_col].copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    for col in data_col:
        if col != timestamp_col:
            df[col] = df[col].astype(float)

    time_intervals_dic = _df_dt_stats(df, date_col=timestamp_col, intervals_dic=None)
    nans_dic = _df_nan_stats(df, perc_rows_less_than=[], perc_rows_more_than=[])

    FILTERING_PARAMS: dict = {"nans": {"efficient_removal": {"preference_tie": "row"}}}

    frequency_minutes = None
    if "frequency_sampling" in dataset_config_dictionary:
        freq_str = dataset_config_dictionary["frequency_sampling"]
        if freq_str:
            assert freq_str in _freq_token_to_minutes, (
                f" frequency_sampling input does not belong to {list(_freq_token_to_minutes.keys())}, "
                "select 'oov' to estimate it from the timestamps"
            )
            frequency_minutes = _freq_token_to_minutes[freq_str]

    if frequency_minutes is None:
        timestamps = pd.to_datetime(df[timestamp_col])
        time_diffs = timestamps.diff().dropna()
        frequency_minutes = float(time_diffs.dt.total_seconds().div(60).median())

    freq_lower = frequency_minutes - _TSFREQUENCY_TOLERANCE * frequency_minutes
    freq_upper = frequency_minutes + _TSFREQUENCY_TOLERANCE * frequency_minutes
    FILTERING_PARAMS["dt"] = {"lower_bound": freq_lower, "upper_bound": freq_upper}

    df = _dq_timeseries_segmentation(
        df, filtering_params=FILTERING_PARAMS, timestamp_tag=timestamp_col
    )

    dataset_config = dataset_config_dictionary.copy()
    dataset_config["id_columns"] = ["segment_id"]
    for col_tag in dataset_config["column_specifiers"]:
        if col_tag not in ("timestamp_column", "autoregressive_modeling"):
            dataset_config["column_specifiers"][col_tag] = [
                c
                for c in dataset_config["column_specifiers"][col_tag]
                if c in df.columns
            ]

    n_minimum = 1
    if task == "inference":
        n_minimum = model_config["context_length"]
    if task == "finetuning":
        n_minimum = model_config["prediction_length"] + model_config["context_length"]

    group_sizes = df.groupby(dataset_config["id_columns"][0]).size()
    large_groups = group_sizes[group_sizes >= n_minimum].index
    df = df[df[dataset_config["id_columns"][0]].isin(large_groups)]

    ts_segments_quality_summary = _time_series_segment_quality_summary(
        df, timestamp_col, dataset_config["id_columns"][0]
    )
    ts_segments_quality_summary["removed_columns"] = [
        c for c in data_col if c not in df.columns
    ]
    ts_segments_quality_summary["frequency_sampling_min"] = frequency_minutes

    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return {
        "data": df,
        "dataset_config_dictionary": dataset_config,
        "dataquality_summary": _make_json_compatible(
            {
                "original_data": {
                    "nans_summary": nans_dic,
                    "sampling_summary": time_intervals_dic,
                },
                "filtered_data_ts_segments": ts_segments_quality_summary,
            }
        ),
    }


# ── TSFM inference helpers ────────────────────────────────────────────────────


def _get_gt_and_predictions(
    trainer, dataset, ix_target_features, inverse_transforms=None
):
    if inverse_transforms is None:
        inverse_transforms = []
    outputs = trainer.predict(dataset)
    target_value_list = []
    pred_value_list = []
    timestamp_id_value_dic: dict = {}
    for i in range(len(dataset)):
        aux = dataset[i]["future_values"][:, ix_target_features].detach().numpy()
        if "timestamp" in dataset[i]:
            timestamp_id_value_dic.setdefault("timestamp", []).append(
                dataset[i]["timestamp"]
            )
        if "id" in dataset[i]:
            timestamp_id_value_dic.setdefault("id", []).extend(list(dataset[i]["id"]))
        target_value_list.append(aux)
        forecast_h = aux.shape[0]
        aux_pred = outputs.predictions[0][
            i, :forecast_h, ix_target_features
        ].transpose()
        pred_value_list.append(aux_pred)
    y_gt = np.array(target_value_list)
    y_pred = np.array(pred_value_list)
    for ix_fhorizon in range(y_gt.shape[1]):
        if inverse_transforms:
            y_gt[:, ix_fhorizon, :] = inverse_transforms[0](y_gt[:, ix_fhorizon, :])
            y_pred[:, ix_fhorizon, :] = inverse_transforms[0](y_pred[:, ix_fhorizon, :])
    return y_gt, y_pred, timestamp_id_value_dic


def _get_performance(
    y_gt,
    y_pred,
    target_columns=None,
    prediction=True,
    inverse_transforms=None,
    ts_mask=None,
):
    if inverse_transforms is None:
        inverse_transforms = []
    if ts_mask is None:
        ts_mask = np.ones([y_gt.shape[0], y_gt.shape[1]])
    if not target_columns:
        target_columns = list(np.arange(y_gt.shape[2]))
    rows = []
    pd_prediction = pd.DataFrame()
    pd_performance = pd.DataFrame()
    for ix_target in range(y_gt.shape[2]):
        for ix_fhorizon in range(y_gt.shape[1]):
            if len(inverse_transforms) > ix_target:
                y_gt[:, ix_fhorizon, ix_target] = inverse_transforms[ix_target](
                    y_gt[:, ix_fhorizon, ix_target][:, np.newaxis]
                )[:, 0]
                y_pred[:, ix_fhorizon, ix_target] = inverse_transforms[ix_target](
                    y_pred[:, ix_fhorizon, ix_target][:, np.newaxis]
                )[:, 0]
            pd_aux = pd.DataFrame(
                {
                    "y_gt": y_gt[:, ix_fhorizon, ix_target],
                    "y_pred": y_pred[:, ix_fhorizon, ix_target],
                    "forecast_horizon": ix_fhorizon + 1,
                    "target": target_columns[ix_target],
                    "on_mask": ts_mask[:, ix_fhorizon],
                }
            )
            pd_prediction = pd.concat([pd_prediction, pd_aux], axis=0)
            y_gt_mask = y_gt[:, ix_fhorizon, ix_target][ts_mask[:, ix_fhorizon] > 0]
            y_pred_mask = y_pred[:, ix_fhorizon, ix_target][ts_mask[:, ix_fhorizon] > 0]
            valid_mask = np.isfinite(y_gt_mask) & np.isfinite(y_pred_mask)
            y_gt_mask = y_gt_mask[valid_mask]
            y_pred_mask = y_pred_mask[valid_mask]
            if y_gt_mask.shape[0] > 0:
                for metric in _METRICS_FORECAST:
                    value = _METRICS_FORECAST[metric](
                        y_gt[:, :ix_fhorizon, ix_target],
                        y_pred[:, :ix_fhorizon, ix_target],
                        axis=1,
                    )
                    stat = np.mean(value) if value is not None else None
                    rows.append(
                        [target_columns[ix_target], ix_fhorizon + 1, metric, stat]
                    )
    if rows:
        pd_performance = pd.DataFrame(
            data=rows, columns=["target", "forecast", "metric", "value"]
        )
    if prediction:
        return pd_performance, pd_prediction
    return pd_performance


def _get_ttm_hf_inference(
    df_dataframe,
    dataset_config_dictionary,
    model_config,
    model_checkpoint,
    scaling=False,
    tsp=None,
    forecast_horizon=-1,
):
    from tsfm_public import TinyTimeMixerForPrediction
    from tsfm_public.toolkit.time_series_preprocessor import (
        TimeSeriesPreprocessor,
        get_datasets,
        create_timestamps,
    )
    from transformers import Trainer, TrainingArguments

    if forecast_horizon == -1:
        forecast_horizon = model_config["prediction_length"]
    else:
        assert forecast_horizon <= model_config["prediction_length"], (
            f" Selected forecast horizon is above what is supported by the model. "
            f"Set a forecast horizon smaller than {model_config['prediction_length']}"
        )
    context_length = model_config["context_length"]
    assert context_length <= len(df_dataframe), (
        " length of dataframe needs to be larger or equal to context length"
    )

    column_specifiers = dataset_config_dictionary["column_specifiers"]
    if (
        "id_columns" in dataset_config_dictionary
        and "id_columns" not in column_specifiers
    ):
        column_specifiers["id_columns"] = dataset_config_dictionary["id_columns"]

    encode_categorical = False
    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        scaling=scaling,
        encode_categorical=encode_categorical,
        prediction_length=forecast_horizon,
        context_length=context_length,
    )
    dataset_dic = get_datasets(
        tsp,
        df_dataframe,
        split_config={"train": 1.0, "test": 0.0},
        use_frequency_token=True,
    )
    dataset_inference = dataset_dic[0]

    model = TinyTimeMixerForPrediction.from_pretrained(
        model_checkpoint, prediction_filter_length=forecast_horizon
    )
    args = TrainingArguments(output_dir="./output", logging_dir="./log")
    trainer = Trainer(model=model, args=args, eval_dataset=dataset_inference)

    ix_target_features = list(
        np.arange(len(dataset_config_dictionary["column_specifiers"]["target_columns"]))
    )

    outputs = trainer.predict(dataset_inference)
    y_pred = outputs.predictions[0][:, :forecast_horizon, ix_target_features]

    if tsp.scaling:
        for ixf in range(y_pred.shape[1]):
            y_pred[:, ixf, :] = tsp.target_scaler_dict["0"].inverse_transform(
                y_pred[:, ixf, :]
            )

    timestamps_list = []
    timestamps_prediction_list = []
    for i in range(len(dataset_inference)):
        if "timestamp" in dataset_inference[i]:
            timestamps_list.append(dataset_inference[i]["timestamp"])
            timestamp_forecast = create_timestamps(
                last_timestamp=dataset_inference[i]["timestamp"],
                time_sequence=df_dataframe[
                    column_specifiers["timestamp_column"]
                ].values,
                periods=forecast_horizon,
            )
            timestamps_prediction_list.append(timestamp_forecast)

    output: dict = {
        "target_columns": dataset_config_dictionary["column_specifiers"][
            "target_columns"
        ],
        "target_prediction": y_pred,
        "timestamp": timestamps_list,
        "timestamp_prediction": timestamps_prediction_list,
    }

    inverse_transforms = []
    if scaling:
        inverse_transforms.append(tsp.target_scaler_dict["0"].inverse_transform)

    y_gt, y_pred_eval, timestamp_id_value_dic = _get_gt_and_predictions(
        trainer,
        dataset_inference,
        ix_target_features=ix_target_features,
        inverse_transforms=inverse_transforms,
    )
    target_columns = dataset_config_dictionary["column_specifiers"]["target_columns"]
    pd_performance = _get_performance(
        y_gt, y_pred_eval, target_columns=target_columns, prediction=False
    )
    output["performance"] = pd_performance

    return output


# ── TSFM finetuning ───────────────────────────────────────────────────────────

_DEFAULT_TRAINING_ARGUMENTS = {
    "overwrite_output_dir": True,
    "learning_rate": 0.0001,
    "num_train_epochs": 10,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "save_strategy": "epoch",
    "logging_strategy": "epoch",
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}


def _ttm_main_config():
    return {
        "scaling": "",
        "p_validation": 0.1,
        "encode_categorical": False,
        "context_length": 512,
        "patch_length": 64,
        "forecast_horizon": 96,
        "batch_size": 32,
        "num_workers": 4,
        "seed": 42,
        "model_type": "ttm",
        "optim": "AdamW",
        "lr": 0.0,
        "epochs": 4,
        "scheduler": "OneCycleLR",
        "epochs_warmup": 5,
        "es_patience": 15.0,
        "es_th": 0.0001,
        "backbone_frozen": False,
        "decoder_mode": "mix_channel",
        "head_dropout": 0.7,
    }


def _finetune_ttm_hf(
    df_dataframe,
    dataset_config_dictionary,
    model_config,
    save_model_dir,
    n_finetune,
    n_calibration,
    n_test,
    model_checkpoint="",
    training_config_dic=None,
):
    from tsfm_public import (
        TinyTimeMixerConfig,
        TinyTimeMixerForPrediction,
        TrackingCallback,
    )
    from tsfm_public.toolkit.lr_finder import optimal_lr_finder
    from tsfm_public.toolkit.time_series_preprocessor import (
        TimeSeriesPreprocessor,
        get_datasets,
    )
    from tsfm_public.toolkit.util import select_by_index
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

    if training_config_dic is None:
        args_config_dic = _ttm_main_config()
    else:
        args_config_dic = training_config_dic.copy()
        default_config = _ttm_main_config()
        for k in default_config:
            if k not in args_config_dic:
                args_config_dic[k] = default_config[k]

    seed = args_config_dic["seed"]
    set_seed(seed)
    encode_categorical = args_config_dic["encode_categorical"]
    scaling_type = args_config_dic["scaling"]
    p_validation = args_config_dic["p_validation"]

    forecast_horizon = model_config["prediction_length"]
    context_length = model_config["context_length"]
    args_config_dic["forecast_horizon"] = forecast_horizon
    args_config_dic["context_length"] = context_length

    assert context_length <= len(df_dataframe), (
        " length of dataframe needs to be >= context length"
    )

    column_specifiers = dataset_config_dictionary["column_specifiers"]
    ix_target_features = list(np.arange(len(column_specifiers["target_columns"])))

    if (
        "id_columns" in dataset_config_dictionary
        and "id_columns" not in column_specifiers
    ):
        column_specifiers["id_columns"] = dataset_config_dictionary["id_columns"]

    n_data = len(df_dataframe)
    assert n_test >= 0
    p_test = n_test / n_data if n_test >= 1 else n_test
    n_train_total = int(np.floor((1 - p_test) * n_data))

    assert n_finetune > 0
    p_finetune = n_finetune / n_train_total if n_finetune > 1 else n_finetune
    n_validation = np.ceil(p_finetune * n_train_total * p_validation)
    p_train = (n_train_total - n_validation) / n_data
    n_train_effective = p_finetune * n_train_total - n_validation
    fewshot_fraction = n_train_effective / (n_train_total - n_validation)

    scaling = scaling_type == "standard"

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        scaling=scaling,
        encode_categorical=encode_categorical,
        prediction_length=forecast_horizon,
        context_length=context_length,
    )
    dataset_dic = get_datasets(
        tsp,
        df_dataframe,
        split_config={"train": p_train, "test": p_test},
        use_frequency_token=True,
        fewshot_fraction=fewshot_fraction,
    )
    train_dataset = dataset_dic[0]
    valid_dataset = dataset_dic[1]
    test_dataset = dataset_dic[2]

    with open(os.path.join(save_model_dir, "args_config.yml"), "w") as outfile:
        yaml.dump(args_config_dic, outfile)
    with open(os.path.join(save_model_dir, "tsp.pickle"), "wb") as _f:
        pickle.dump(tsp, _f)

    if os.path.exists(model_checkpoint):
        finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
            model_checkpoint,
            head_dropout=args_config_dic["head_dropout"],
            num_input_channels=tsp.num_input_channels,
            exogenous_channel_indices=tsp.exogenous_channel_indices,
            prediction_channel_indices=tsp.prediction_channel_indices,
            decoder_mode=args_config_dic["decoder_mode"],
            enable_forecast_channel_mixing=False,
            fcm_use_mixer=False,
            ignore_mismatched_sizes=True,
            prediction_filter_length=forecast_horizon,
        )
    else:
        config_ttm_dic = model_config.copy()
        config_ttm_dic.update(
            {
                "head_dropout": args_config_dic["head_dropout"],
                "prediction_length": forecast_horizon,
                "num_input_channels": tsp.num_input_channels,
                "exogenous_channel_indices": tsp.exogenous_channel_indices,
                "prediction_channel_indices": tsp.prediction_channel_indices,
                "enable_forecast_channel_mixing": False,
                "fcm_use_mixer": False,
                "decoder_mode": args_config_dic["decoder_mode"],
            }
        )
        config = TinyTimeMixerConfig(**config_ttm_dic)
        finetune_forecast_model = TinyTimeMixerForPrediction(config)

    if args_config_dic["backbone_frozen"]:
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False

    batch_size = args_config_dic["batch_size"]
    epochs = args_config_dic["epochs"]
    num_workers = args_config_dic["num_workers"]
    epochs_warmup = args_config_dic["epochs_warmup"]
    es_patience = args_config_dic["es_patience"]
    es_th = args_config_dic["es_th"]
    optim = args_config_dic["optim"]
    scheduler = args_config_dic["scheduler"]
    lr = args_config_dic["lr"]

    # Use a fresh copy of the defaults to avoid cross-call mutation
    training_config_dictionary = _DEFAULT_TRAINING_ARGUMENTS.copy()

    output_fewshot_dir = save_model_dir + "/fewshot/"
    logging_dir = save_model_dir + "/log/"
    os.makedirs(output_fewshot_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    training_config_dictionary.update(
        {
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": epochs,
            "learning_rate": lr,
            "output_dir": output_fewshot_dir,
            "logging_dir": logging_dir,
            "dataloader_num_workers": num_workers,
        }
    )
    if epochs_warmup > 0:
        training_config_dictionary["warmup_steps"] = math.ceil(
            epochs_warmup * len(train_dataset) / batch_size
        )
    with open(os.path.join(save_model_dir, "training_config.yml"), "w") as outfile:
        yaml.dump(training_config_dictionary, outfile)

    finetune_forecast_args = TrainingArguments(**training_config_dictionary)

    if n_finetune > 0:
        if lr <= 0:
            try:
                lr, finetune_forecast_model = optimal_lr_finder(
                    finetune_forecast_model, train_dataset, batch_size=batch_size
                )
                if lr <= 0:
                    lr = 0.0001
            except Exception:
                lr = 0.0001
    else:
        lr = 0.0001

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=es_patience,
        early_stopping_threshold=es_th,
    )

    optimizer = None
    if optim == "AdamW":
        from torch.optim import AdamW

        optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)

    scheduler_object = None
    if scheduler == "cosine_with_warmup":
        if optimizer is None:
            from torch.optim import AdamW

            optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)
        from transformers.optimization import get_cosine_schedule_with_warmup

        total_steps = math.ceil(len(train_dataset) * epochs / batch_size)
        num_warmup_steps = math.ceil(epochs_warmup * len(train_dataset) / batch_size)
        scheduler_object = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )
    if scheduler == "OneCycleLR":
        if optimizer is None:
            from torch.optim import AdamW

            optimizer = AdamW(finetune_forecast_model.parameters(), lr=lr)
        from torch.optim.lr_scheduler import OneCycleLR

        scheduler_object = OneCycleLR(
            optimizer,
            lr,
            epochs=epochs,
            steps_per_epoch=math.ceil(len(train_dataset) / batch_size),
        )

    tracking_callback = TrackingCallback()
    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler_object),
    )

    start_time = time.time()
    if n_finetune > 0:
        finetune_forecast_trainer.train()
    train_time = time.time() - start_time

    dataset_eval: dict = {}
    if n_finetune > 0:
        dataset_eval["train"] = train_dataset
        dataset_eval["valid"] = valid_dataset
    if n_test >= 1:
        dataset_eval["test"] = test_dataset

    pd_performance = pd.DataFrame()
    for dataset_key in dataset_eval:
        inverse_transforms_eval = []
        if scaling:
            inverse_transforms_eval.append(
                tsp.target_scaler_dict["0"].inverse_transform
            )
        y_gt, y_pred_eval, _ = _get_gt_and_predictions(
            finetune_forecast_trainer,
            dataset_eval[dataset_key],
            ix_target_features=ix_target_features,
            inverse_transforms=inverse_transforms_eval,
        )
        target_columns = dataset_config_dictionary["column_specifiers"][
            "target_columns"
        ]
        pd_performance_i = _get_performance(
            y_gt, y_pred_eval, target_columns=target_columns, prediction=False
        )
        pd_performance_i["split"] = dataset_key
        pd_performance = pd.concat([pd_performance, pd_performance_i], axis=0)

    pd_performance["train_time"] = train_time
    return {
        "performance": pd_performance,
        "save_model_dir": save_model_dir,
        "experiment_config_path": os.path.join(save_model_dir, "args_config.yml"),
    }


def _find_largest_tsfm_checkpoint_directory(root_dir: str) -> str:
    largest_checkpoint_dir = None
    largest_number = float("-inf")
    for f in os.listdir(root_dir):
        if "checkpoint" in f:
            number = int(f.split("-")[-1])
            if number > largest_number:
                largest_number = number
                largest_checkpoint_dir = os.path.join(root_dir, f)
    return largest_checkpoint_dir


# ── Conformal anomaly detection ───────────────────────────────────────────────

_NONCONFORMITY_SCORES = ["absolute_error"]


def _absolute_error(y, y_pred):
    assert y.shape == y_pred.shape, (
        f"y and y_pred shapes do not match: {y.shape} vs {y_pred.shape}"
    )
    error = np.abs(y - y_pred)
    if len(error.shape) > 1:
        error = np.mean(error, axis=-1)
    return error


def _nonconformity_score_functions(
    y_pred, y_gt, X=None, nonconformity_score="absolute_error"
):
    assert nonconformity_score in _NONCONFORMITY_SCORES
    if nonconformity_score == "absolute_error":
        return _absolute_error(y_gt, y_pred)


def _conformal_set(y_pred, score_threshold, nonconformity_score="absolute_error"):
    if nonconformity_score == "absolute_error":
        return {"y_low": y_pred - score_threshold, "y_high": y_pred + score_threshold}


def _weighted_conformal_quantile(
    scores, weights, alpha=0.05, conformal_correction=False, max_score=np.inf
):
    if weights is None:
        weights = np.ones_like(scores)
    assert np.max(weights) <= 1
    assert np.min(weights) >= 0
    assert weights.shape[0] == scores.shape[0]
    if conformal_correction:
        weights = np.append(weights, np.array([1]))
        scores = np.append(scores, np.array([max_score]))
    weights = np.array(weights) / np.sum(weights)
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    quantile_index = np.searchsorted(cumulative_weights, 1 - alpha)
    return sorted_scores[quantile_index]


def _weighted_conformal_alpha(
    scores, weights, score_observed, conformal_correction=False, max_score=np.inf
):
    if weights is None:
        weights = np.ones_like(score_observed)
    if conformal_correction:
        weights = np.append(weights, np.array([1]))
        scores = np.append(scores, np.array([max_score]))
    weights = np.array(weights) / np.sum(weights)
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    return np.sum(sorted_weights[sorted_scores > score_observed])


class _TSADWeightedConformalWrapper:
    def __init__(
        self,
        nonconformity_score="absolute_error",
        false_alarm=0.05,
        weighting="uniform",
        weighting_params=None,
        threshold_function="weighting",
        window_size=None,
        online_adaptive=False,
    ):
        if weighting_params is None:
            weighting_params = {}
        self.nonconformity_score = nonconformity_score
        assert self.nonconformity_score in _NONCONFORMITY_SCORES
        self.nonconformity_score_func = _nonconformity_score_functions
        self.quantile = 1 - false_alarm
        self.false_alarm = false_alarm
        self.weighting = weighting
        self.weighting_params = weighting_params
        self.window_size = window_size
        self.online_size = 1
        self.online = online_adaptive
        self.threshold_function = threshold_function
        self.cal_scores: list = []
        self.weights: list = []
        self.cal_X: list = []
        self.cal_timestamps: list = []
        self.score_threshold = None

    def fit(self, y_cal_pred, y_cal_gt, X_cal=None, cal_timestamps=None):
        if self.window_size is None:
            self.window_size = y_cal_pred.shape[0]
        self.cal_scores = _nonconformity_score_functions(
            y_cal_pred, y_cal_gt, X=X_cal, nonconformity_score=self.nonconformity_score
        )
        self.cal_scores = self.cal_scores[-self.window_size :]
        if X_cal is not None:
            self.cal_X = X_cal[-self.window_size :]
        if cal_timestamps is not None:
            self.cal_timestamps = cal_timestamps[-self.window_size :]
        if self.weighting in ("uniform", "exponential_decay"):
            cal_weights = self.get_weights()
            self.weights.append(cal_weights)
            if self.threshold_function == "weighting":
                self.score_threshold = self._score_threshold_func(
                    cal_weights, false_alarm=self.false_alarm
                )
        critical_efficient_size = np.ceil(1 / self.false_alarm)
        assert np.sum(cal_weights) >= critical_efficient_size, (
            f" The effective size is too small for the desired false alarm of {self.false_alarm}, "
            f"the calibration set should be larger than {critical_efficient_size}"
        )

    def get_weights(self, y_pred=None, X=None, timestamps=None, false_alarm=None):
        if false_alarm is None:
            false_alarm = self.false_alarm
        if self.weighting in ("uniform", "exponential_decay"):
            if self.weights:
                return self.weights[-1]
            if self.weighting == "uniform":
                return np.ones(self.window_size)
            if self.weighting == "exponential_decay":
                decay_param = self.weighting_params.get("decay_param", 0.99)
                return decay_param ** (self.window_size - np.arange(self.window_size))

    def _score_threshold_func(
        self,
        cal_weights,
        cal_scores=None,
        y_pred=None,
        X=None,
        timestamps=None,
        false_alarm=None,
    ):
        if cal_scores is None:
            cal_scores = self.cal_scores
        if false_alarm is None:
            false_alarm = self.false_alarm
        score_threshold = []
        if self.threshold_function == "weighting":
            if len(cal_weights.shape) == 1:
                score_threshold = _weighted_conformal_quantile(
                    np.append(cal_scores, np.array([np.infty]), axis=0),
                    np.append(cal_weights, np.array([1]), axis=0),
                    alpha=false_alarm,
                )
            else:
                for i in range(cal_weights.shape[0]):
                    st_i = _weighted_conformal_quantile(
                        np.append(cal_scores, np.array([np.infty]), axis=0),
                        np.append(cal_weights[i, :], np.array([1]), axis=0),
                        alpha=false_alarm,
                    )
                    score_threshold.append(st_i)
                score_threshold = np.array(score_threshold)
        return score_threshold

    def predict_batch(
        self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=None
    ):
        if false_alarm is None:
            false_alarm = self.false_alarm
        if update is None:
            update = self.online
        cal_weights = self.get_weights(
            y_pred, X=X, timestamps=timestamps, false_alarm=false_alarm
        )
        if (
            false_alarm == self.false_alarm
            and self.weighting in ("uniform",)
            and self.threshold_function in ("weighting",)
        ):
            score_threshold = self.score_threshold
        else:
            score_threshold = self._score_threshold_func(
                cal_weights,
                y_pred=y_pred,
                X=X,
                timestamps=timestamps,
                false_alarm=false_alarm,
            )
        prediction_interval = _conformal_set(
            y_pred, score_threshold, nonconformity_score=self.nonconformity_score
        )
        output: dict = {}
        if y_gt is not None:
            test_scores = _nonconformity_score_functions(
                y_pred, y_gt, X=X, nonconformity_score=self.nonconformity_score
            )
            test_outliers = np.array(test_scores > score_threshold).astype("int")
            test_ad_scores = [
                _weighted_conformal_alpha(
                    np.append(self.cal_scores, np.array([np.infty]), axis=0),
                    np.append(cal_weights, np.array([1]), axis=0),
                    score,
                )
                for score in test_scores
            ]
            if update:
                self.update(test_scores, X=X, timestamps=timestamps)
            output["outliers"] = test_outliers
            output["outliers_scores"] = np.array(test_ad_scores).flatten()
        output["prediction_interval"] = prediction_interval
        return output

    def predict(
        self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=None
    ):
        if false_alarm is None:
            false_alarm = self.false_alarm
        if update is None:
            update = self.online
        n_samples = y_pred.shape[0]
        n_batches = int(np.ceil(n_samples / self.online_size))
        if y_gt is not None and update:
            output = None
            for ix_b in range(n_batches):
                ix_ini = int(ix_b * self.online_size)
                ix_end = min(
                    int(ix_b * self.online_size + self.online_size), y_pred.shape[0]
                )
                y_pred_b = y_pred[ix_ini:ix_end]
                y_gt_b = y_gt[ix_ini:ix_end]
                X_b = X[ix_ini:ix_end] if X is not None else None
                ts_b = timestamps[ix_ini:ix_end] if timestamps is not None else None
                output_b = self.predict_batch(
                    y_pred_b,
                    y_gt=y_gt_b,
                    X=X_b,
                    timestamps=ts_b,
                    false_alarm=false_alarm,
                    update=update,
                )
                if output is None:
                    output = output_b.copy()
                else:
                    for k in output_b:
                        if k == "prediction_interval":
                            for k2 in output_b[k]:
                                output[k][k2] = np.append(
                                    output[k][k2], np.array(output_b[k][k2]), axis=0
                                )
                        else:
                            output[k] = np.append(
                                output[k], np.array(output_b[k]), axis=0
                            )
        else:
            output = self.predict_batch(
                y_pred,
                y_gt=y_gt,
                X=X,
                timestamps=timestamps,
                false_alarm=false_alarm,
                update=update,
            )
        return output

    def update(self, scores, X=None, timestamps=None):
        self.cal_scores = np.append(self.cal_scores, scores, axis=0)
        self.cal_scores = self.cal_scores[-self.window_size :]
        if timestamps is not None:
            self.cal_timestamps.extend(timestamps)
            self.cal_timestamps = self.cal_timestamps[-self.window_size :]
        if X is not None:
            self.cal_X = np.append(self.cal_X, X, axis=0)
            self.cal_X = self.cal_X[-self.window_size :]
        if self.weighting == "uniform":
            cal_weights = self.get_weights()
            if self.threshold_function == "weighting":
                self.score_threshold = self._score_threshold_func(
                    cal_weights, false_alarm=self.false_alarm
                )


# ── TSAD data alignment ───────────────────────────────────────────────────────


def _get_tsfm_dataloaders(
    df_dataframe, model_config, dataset_config_dictionary, scaling=False
):
    from tsfm_public.toolkit.dataset import ForecastDFDataset
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
    from tsfm_public.toolkit.util import select_by_index

    forecast_horizon = model_config["prediction_length"]
    context_length = model_config["context_length"]
    assert context_length <= len(df_dataframe), (
        " length of dataframe needs to be >= context length"
    )

    column_specifiers = dataset_config_dictionary["column_specifiers"]
    id_columns = dataset_config_dictionary.get("id_columns", [])

    data = select_by_index(
        df_dataframe, id_columns=[], start_index=0, end_index=len(df_dataframe)
    )

    tsp = TimeSeriesPreprocessor(
        **column_specifiers, scaling=scaling, encode_categorical=False
    )
    tsp = tsp.train(data)

    dataset_inference = ForecastDFDataset(
        tsp.preprocess(data),
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_horizon,
        id_columns=id_columns,
    )
    return dataset_inference


def _tsfm_dataloader_to_array(
    dataset_calibration, ix_target_features, x_context_window=-1
):
    y_gt = []
    X = []
    timestamp_id_value_dic: dict = {}
    for i in range(len(dataset_calibration)):
        try:
            y_gt.append(
                dataset_calibration[i]["future_values"][:, ix_target_features]
                .detach()
                .numpy()
            )
        except Exception as ex:
            raise ValueError(
                f"At least one of the target_columns is not in the input file: {ix_target_features}"
            ) from ex
        X.append(dataset_calibration[i]["past_values"].detach().numpy())
        if "timestamp" in dataset_calibration[i]:
            timestamp_id_value_dic.setdefault("timestamp", []).append(
                dataset_calibration[i]["timestamp"]
            )
        if "id" in dataset_calibration[i]:
            timestamp_id_value_dic.setdefault("id", []).extend(
                list(dataset_calibration[i]["id"])
            )
    y_gt_arr = np.array(y_gt)
    if len(y_gt_arr.shape) > 1:
        y_gt_arr = y_gt_arr[:, 0]
    y_gt_arr = np.squeeze(y_gt_arr)
    X_arr = np.array(X)
    if x_context_window > 0:
        X_arr = X_arr[:, -int(x_context_window) :, :]
    X_arr = X_arr.reshape([X_arr.shape[0], -1])
    return X_arr, y_gt_arr, timestamp_id_value_dic


def _get_tsad_aligned_data(
    df_data, dataset_config, ad_config, tsmodel_prediction_dictionary
):
    from tsfm_public.toolkit.time_series_preprocessor import create_timestamps

    context_length = ad_config["context_length"]
    prediction_length = ad_config["prediction_length"]
    scaling = ad_config["scaling"]
    ix_target_features = list(
        np.arange(len(dataset_config["column_specifiers"]["target_columns"]))
    )

    df_data[dataset_config["column_specifiers"]["timestamp_column"]] = pd.to_datetime(
        df_data[dataset_config["column_specifiers"]["timestamp_column"]]
    )
    dataset_inference = _get_tsfm_dataloaders(
        df_data,
        {"prediction_length": prediction_length, "context_length": context_length},
        dataset_config,
        scaling=scaling,
    )
    X, y_gt, timestamp_id_value_dic = _tsfm_dataloader_to_array(
        dataset_inference, ix_target_features, x_context_window=context_length
    )

    source_timestamp = np.array(tsmodel_prediction_dictionary["timestamp"])[:, 0]
    target_timestamp = timestamp_id_value_dic["timestamp"]

    forecast_horizon = 1
    target_timestamp_updated = []
    for ts in target_timestamp:
        ts_updated = create_timestamps(
            last_timestamp=ts,
            time_sequence=target_timestamp,
            periods=forecast_horizon,
        )[0]
        target_timestamp_updated.append(ts_updated)
    target_timestamp = np.array(
        np.array(target_timestamp_updated, dtype="datetime64[ns]")
    )
    source_timestamp = np.array(np.array(source_timestamp, dtype="datetime64[ns]"))

    frequency_sampling_median = np.median(target_timestamp[1:] - target_timestamp[:-1])
    tolerance_frequency_sampling = 0.2

    time_diff = np.abs(target_timestamp[:, None] - source_timestamp)
    matching_pairs = np.where(
        time_diff <= frequency_sampling_median * tolerance_frequency_sampling
    )
    index_timestamp = matching_pairs[0]
    index_timestamp_source = matching_pairs[1]

    X_cp = X[index_timestamp]
    y_gt_cp = y_gt[index_timestamp]
    y_pred = np.array(tsmodel_prediction_dictionary["target_prediction"])[
        index_timestamp_source, 0, 0
    ]
    timestamps_source = np.array(source_timestamp)[index_timestamp_source]

    return {
        "X": X_cp,
        "y_gt": y_gt_cp,
        "y_pred": y_pred,
        "timestamp": timestamps_source,
    }


# ── TSAD orchestration ────────────────────────────────────────────────────────

_AD_CONFIG_DEFAULT = {
    "ad_model_type": "timeseries_conformal",
    "context_length": 1,
    "false_alarm": 0.01,
    "window_size": None,
    "weighting": "uniform",
    "weighting_params": {},
}


class _TimeSeriesAnomalyDetectionConformalWrapper:
    def run(
        self,
        dataset_path: str,
        dataset_config_dictionary: dict,
        tsmodel_prediction_dictionary: dict,
        ad_model_checkpoint: str = None,
        ad_model_save: str = None,
        task: str = "inference",
        ad_model_type: str = None,
        n_calibration=None,
        false_alarm: float = None,
        context_length: int = None,
    ) -> dict:
        ad_model = None
        if ad_model_checkpoint is not None:
            if os.path.exists(ad_model_checkpoint):
                assert os.path.exists(ad_model_checkpoint + "/model.pkl")
                assert os.path.exists(ad_model_checkpoint + "/config.json")
                with open(ad_model_checkpoint + "/model.pkl", "rb") as _f:
                    ad_model = pickle.load(_f)
                with open(ad_model_checkpoint + "/config.json") as _f:
                    ad_config = json.load(_f)
                ad_model_type = ad_config["ad_model_type"]
                context_length = ad_config["context_length"]
                if false_alarm is None:
                    false_alarm = ad_config["false_alarm"]
                elif ad_config["false_alarm"] != false_alarm:
                    if task != "fit":
                        false_alarm = ad_config["false_alarm"]
        else:
            ad_config = {
                "context_length": context_length
                if context_length is not None
                else _AD_CONFIG_DEFAULT["context_length"],
                "false_alarm": false_alarm
                if false_alarm is not None
                else _AD_CONFIG_DEFAULT["false_alarm"],
                "ad_model_type": ad_model_type
                if ad_model_type is not None
                else _AD_CONFIG_DEFAULT["ad_model_type"],
            }
            context_length = ad_config["context_length"]
            false_alarm = ad_config["false_alarm"]
            ad_model_type = ad_config["ad_model_type"]

        df_data = _read_ts_data(
            dataset_path, dataset_config_dictionary=dataset_config_dictionary
        )
        context_length = ad_config["context_length"]
        output_tsad_aligned = _get_tsad_aligned_data(
            df_data,
            dataset_config_dictionary,
            ad_config={
                "prediction_length": 1,
                "context_length": context_length,
                "scaling": False,
            },
            tsmodel_prediction_dictionary=tsmodel_prediction_dictionary,
        )

        timestamps_source = output_tsad_aligned["timestamp"]
        X_cp = output_tsad_aligned["X"]
        y_gt_cp = output_tsad_aligned["y_gt"]
        y_pred = output_tsad_aligned["y_pred"]

        output_ad: dict = {}
        if task == "fit":
            if n_calibration is None:
                n_calibration = y_pred.shape[0]
            if n_calibration < 1:
                n_calibration = int(np.ceil(y_pred.shape[0] * n_calibration))
            n_calibration = int(n_calibration)
            n_critical = int(np.ceil(1 / false_alarm))
            assert n_calibration >= n_critical, (
                f" n_calibration should be >= {n_critical}, "
                f"otherwise increase false alarm to {round(1 / n_calibration, 2)}"
            )

            X_cp_cal = X_cp[:n_calibration]
            y_gt_cp_cal = y_gt_cp[:n_calibration]
            y_pred_cal = y_pred[:n_calibration]

            if ad_model_type in (
                "timeseries_conformal",
                "timeseries_conformal_adaptive",
            ):
                update = ad_model_type == "timeseries_conformal_adaptive"
                ad_model = _TSADWeightedConformalWrapper(
                    false_alarm=false_alarm,
                    weighting=_AD_CONFIG_DEFAULT["weighting"],
                    window_size=_AD_CONFIG_DEFAULT["window_size"],
                    weighting_params=_AD_CONFIG_DEFAULT["weighting_params"],
                    online_adaptive=update,
                )
                ad_model.fit(
                    y_cal_pred=np.array(y_pred_cal), y_cal_gt=np.array(y_gt_cp_cal)
                )
                output_prediction = ad_model.predict(
                    y_pred=np.array(y_pred), y_gt=np.array(y_gt_cp), update=update
                )
                output_ad = {
                    "timestamp": timestamps_source,
                    "KPI": tsmodel_prediction_dictionary["target_columns"],
                    "value": np.array(y_gt_cp),
                    "upper_bound": output_prediction["prediction_interval"]["y_high"],
                    "lower_bound": output_prediction["prediction_interval"]["y_low"],
                    "anomaly_score": 1 - output_prediction["outliers_scores"],
                    "anomaly_label": output_prediction["outliers"] == 1,
                    "split": [
                        "calibration" if i < n_calibration else "test"
                        for i in range(y_pred.shape[0])
                    ],
                }

            if ad_model is not None and ad_model_save is not None:
                with open(ad_model_save + "/model.pkl", "wb") as _f:
                    pickle.dump(ad_model, _f)
                with open(ad_model_save + "/config.json", "w") as _f:
                    json.dump(ad_config, _f)

        if task == "inference":
            if false_alarm is None:
                false_alarm = ad_model.false_alarm
            output_prediction = ad_model.predict(
                y_pred=np.array(y_pred), y_gt=np.array(y_gt_cp)
            )
            output_ad = {
                "timestamp": timestamps_source,
                "KPI": tsmodel_prediction_dictionary["target_columns"],
                "value": np.array(y_gt_cp),
                "upper_bound": output_prediction["prediction_interval"]["y_high"],
                "lower_bound": output_prediction["prediction_interval"]["y_low"],
                "anomaly_score": 1 - output_prediction["outliers_scores"],
                "anomaly_label": output_prediction["outliers"] == 1,
                "split": ["test"] * output_prediction["outliers_scores"].shape[0],
            }

        return output_ad



# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_dataset_config(
    timestamp_column: str,
    target_columns: List[str],
    conditional_columns: Optional[List[str]],
    id_columns: Optional[List[str]],
    frequency_sampling: str,
    autoregressive_modeling: bool,
) -> dict:
    return {
        "column_specifiers": {
            "autoregressive_modeling": autoregressive_modeling,
            "timestamp_column": timestamp_column,
            "conditional_columns": conditional_columns or [],
            "target_columns": target_columns,
        },
        "id_columns": id_columns or [],
        "frequency_sampling": frequency_sampling,
    }


def _tsad_output_to_df(output: dict) -> pd.DataFrame:
    kpi = output.pop("KPI", None)
    df = pd.DataFrame.from_dict({k: np.array(v) for k, v in output.items()})
    if kpi is not None:
        df["KPI"] = (
            kpi[0] if (hasattr(kpi, "__len__") and not isinstance(kpi, str)) else kpi
        )
    return df


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("TSFMAgent")


# ── Static tools ──────────────────────────────────────────────────────────────


@mcp.tool()
def get_ai_tasks() -> AITasksResult:
    """Returns the list of supported AI task types for time-series analysis.

    Tasks: tsfm_integrated_tsad, tsfm_forecasting, tsfm_forecasting_finetune,
    tsfm_forecasting_evaluation.
    """
    return AITasksResult(tasks=[AITaskEntry(**t) for t in _AI_TASKS])


@mcp.tool()
def get_tsfm_models() -> TSFMModelsResult:
    """Returns the list of available pre-trained TinyTimeMixer (TTM) model checkpoints.

    Models: ttm_96_28 (context_length=96), ttm_512_96 (context_length=512),
    and energy-domain fine-tuned variants of both.
    """
    return TSFMModelsResult(models=[TSFMModelEntry(**m) for m in _TSFM_MODELS])


# ── TSFM Forecasting (zero-shot inference) ────────────────────────────────────


@mcp.tool()
def run_tsfm_forecasting(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    model_checkpoint: str = "ttm_96_28",
    forecast_horizon: int = -1,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: str = "oov",
    autoregressive_modeling: bool = True,
    include_dataquality_summary: bool = False,
) -> Union[ForecastingResult, ErrorResult]:
    """Run zero-shot time-series forecasting with a TinyTimeMixer (TTM) model.

    Returns a ForecastingResult whose results_file is the path to a JSON file
    with raw predictions (target_prediction, timestamp, target_columns arrays).
    Pass that path to run_tsad as tsfm_output_json for anomaly detection.

    Args:
        dataset_path: Path to a CSV, JSON, or XLSX dataset.
        timestamp_column: Name of the timestamp column.
        target_columns: Columns to forecast.
        model_checkpoint: Model name (e.g. ttm_96_28) or absolute checkpoint path.
        forecast_horizon: Number of steps to forecast; -1 uses the model default.
        conditional_columns: Exogenous / conditional feature columns.
        id_columns: ID columns for multi-entity grouped time series.
        frequency_sampling: Sampling frequency string (e.g. '15_minutes') or
            'oov' to auto-detect from the data.
        autoregressive_modeling: Use autoregressive inference when True.
        include_dataquality_summary: Attach a data-quality report to the result.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")

    try:
        import tsfm_public  # noqa: F401 – verify dependency present
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    model_checkpoint = _get_model_checkpoint_path(model_checkpoint)
    dataset_path = _get_dataset_path(dataset_path)
    dataset_config = _build_dataset_config(
        timestamp_column,
        target_columns,
        conditional_columns,
        id_columns,
        frequency_sampling,
        autoregressive_modeling,
    )

    try:
        data_df = _read_ts_data(dataset_path, dataset_config_dictionary=dataset_config)
        with open(model_checkpoint + "/config.json") as _f:
            model_config = json.load(_f)

        output_data_quality = _tsfm_data_quality_filter(
            data_df, dataset_config, model_config, task="inference"
        )
        data_df = output_data_quality["data"]
        dataset_config = output_data_quality["dataset_config_dictionary"]

        inference_result_dict_data: dict = {
            "target_prediction": [],
            "timestamp": [],
            "target_columns": [],
        }

        if len(data_df) > 0:
            output = _get_ttm_hf_inference(
                data_df,
                dataset_config,
                model_config,
                model_checkpoint,
                forecast_horizon=forecast_horizon,
            )
            inference_result_dict_data["target_prediction"] = output[
                "target_prediction"
            ].tolist()
            inference_result_dict_data["timestamp"] = (
                np.array(output["timestamp_prediction"]).astype(str).tolist()
            )
            inference_result_dict_data["target_columns"] = output["target_columns"]
        else:
            return ErrorResult(
                error="Data quality was poor; after filtering, no continuous segment satisfied the "
                "context length requirement. Check Data Quality Summary."
            )

        # Trim to requested forecast horizon
        if forecast_horizon != -1 and "target_prediction" in inference_result_dict_data:
            target_prediction = np.array(
                inference_result_dict_data["target_prediction"]
            )
            if 0 < forecast_horizon <= target_prediction.shape[1]:
                inference_result_dict_data["target_prediction"] = target_prediction[
                    :, :forecast_horizon, :
                ].tolist()
                inference_result_dict_data["timestamp"] = np.array(
                    inference_result_dict_data["timestamp"]
                )[:, :forecast_horizon].tolist()

        results_file = _write_json_to_temp(
            json.dumps(inference_result_dict_data, indent=4)
        )

    except Exception as exc:
        logger.error("run_tsfm_forecasting failed: %s", exc)
        return ErrorResult(error=str(exc))

    dataquality_summary = (
        output_data_quality["dataquality_summary"]
        if include_dataquality_summary
        else None
    )
    return ForecastingResult(
        status="success",
        results_file=results_file,
        dataquality_summary=dataquality_summary,
        message=f"Forecasting complete. Predictions saved to {results_file}.",
    )


# ── TSFM Finetuning ───────────────────────────────────────────────────────────


@mcp.tool()
def run_tsfm_finetuning(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    model_checkpoint: str = "ttm_96_28",
    save_model_dir: str = "tuned_models",
    forecast_horizon: int = -1,
    n_finetune: float = 0.05,
    n_calibration: float = 0.0,
    n_test: float = 0.05,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: str = "oov",
    autoregressive_modeling: bool = True,
    include_dataquality_summary: bool = False,
) -> Union[FinetuningResult, ErrorResult]:
    """Few-shot fine-tune a TinyTimeMixer model on a local dataset.

    Returns a FinetuningResult with the saved checkpoint path and a JSON file
    containing per-forecast-horizon performance metrics.

    Args:
        dataset_path: Path to the training dataset (CSV/JSON/XLSX).
        timestamp_column: Name of the timestamp column.
        target_columns: Columns to forecast and fine-tune on.
        model_checkpoint: Base model to start from (e.g. ttm_96_28).
        save_model_dir: Directory to save the fine-tuned model checkpoint.
        forecast_horizon: Steps to forecast; -1 uses the model default.
        n_finetune: Fraction (≤1) or count (>1) of samples for fine-tuning.
        n_calibration: Fraction or count for calibration set (default 0).
        n_test: Fraction or count for test evaluation (default 0.05).
        conditional_columns: Exogenous feature columns.
        id_columns: ID columns for grouped time series.
        frequency_sampling: Sampling frequency string or 'oov' to auto-detect.
        autoregressive_modeling: Use autoregressive mode when True.
        include_dataquality_summary: Attach a data-quality report to the result.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")

    try:
        import tsfm_public  # noqa: F401
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    model_checkpoint = _get_model_checkpoint_path(model_checkpoint)
    dataset_path = _get_dataset_path(dataset_path)
    abs_save_dir = _get_outputs_path(save_model_dir)
    dataset_config = _build_dataset_config(
        timestamp_column,
        target_columns,
        conditional_columns,
        id_columns,
        frequency_sampling,
        autoregressive_modeling,
    )

    try:
        data_df = _read_ts_data(dataset_path, dataset_config_dictionary=dataset_config)
        with open(model_checkpoint + "/config.json") as _f:
            model_config = json.load(_f)

        os.makedirs(abs_save_dir, exist_ok=True)

        output_data_quality = _tsfm_data_quality_filter(
            data_df, dataset_config, model_config, task="finetuning"
        )
        data_df = output_data_quality["data"]
        dataset_config = output_data_quality["dataset_config_dictionary"]

        if len(data_df) == 0:
            return ErrorResult(
                error="Data quality was poor; after filtering, no continuous segment satisfied the "
                "context length requirement. Check Data Quality Summary."
            )

        output = _finetune_ttm_hf(
            data_df,
            dataset_config,
            model_config,
            abs_save_dir,
            n_finetune,
            n_calibration,
            n_test,
            model_checkpoint=model_checkpoint,
        )

        result_dict = output.copy()
        result_dict["performance"] = result_dict["performance"].to_dict()

        if "performance" in result_dict:
            df_perf = pd.DataFrame(result_dict["performance"])
            df_perf["forecast"] = df_perf["forecast"].values + 1
            max_forecast = df_perf["forecast"].max()
            if 0 < forecast_horizon <= max_forecast:
                result_dict["performance"] = df_perf.loc[
                    df_perf["forecast"] == forecast_horizon
                ].to_dict()

        if include_dataquality_summary:
            result_dict["dataquality_summary"] = output_data_quality[
                "dataquality_summary"
            ]

        results_file = _write_json_to_temp(json.dumps(result_dict, indent=4))

    except Exception as exc:
        logger.error("run_tsfm_finetuning failed: %s", exc)
        return ErrorResult(error=str(exc))

    try:
        fewshot_dir = abs_save_dir + "/fewshot/"
        saved_checkpoint = (
            _find_largest_tsfm_checkpoint_directory(fewshot_dir) or abs_save_dir
        ) + "/"
    except Exception as exc:
        logger.warning("Could not resolve finetuned checkpoint path: %s", exc)
        saved_checkpoint = save_model_dir

    return FinetuningResult(
        status="success",
        model_checkpoint=saved_checkpoint,
        results_file=results_file,
        message=(
            f"Fine-tuning complete. Model saved to {saved_checkpoint}. "
            f"Metrics saved to {results_file}."
        ),
    )


# ── TSAD (conformal anomaly detection on top of TSFM forecasts) ──────────────


@mcp.tool()
def run_tsad(
    dataset_path: str,
    tsfm_output_json: str,
    timestamp_column: str,
    target_columns: List[str],
    task: str = "fit",
    false_alarm: float = 0.05,
    ad_model_type: str = "timeseries_conformal_adaptive",
    ad_model_checkpoint: Optional[str] = None,
    ad_model_save: Optional[str] = None,
    n_calibration: float = 0.2,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: Optional[str] = None,
    autoregressive_modeling: bool = True,
) -> Union[TSADResult, ErrorResult]:
    """Run conformal anomaly detection on TSFM forecasting output.

    tsfm_output_json must be the results_file path returned by run_tsfm_forecasting.
    Fits (or loads) a conformal AD model and saves anomaly-labelled predictions to CSV.

    Args:
        dataset_path: Path to the original time-series dataset.
        tsfm_output_json: Path to JSON predictions file from run_tsfm_forecasting.
        timestamp_column: Name of the timestamp column.
        target_columns: Target columns that were forecast.
        task: 'fit' to train a new AD model, 'inference' to use an existing one.
        false_alarm: False alarm rate (1 − coverage); default 0.05 → 95% coverage.
        ad_model_type: 'timeseries_conformal' or 'timeseries_conformal_adaptive'.
        ad_model_checkpoint: Path to an existing AD model (required for 'inference').
        ad_model_save: Directory to save the fitted AD model.
        n_calibration: Fraction of data used for calibration (default 0.2).
        conditional_columns: Exogenous feature columns.
        id_columns: ID columns for grouped time series.
        frequency_sampling: Sampling frequency string or None to auto-detect.
        autoregressive_modeling: Use autoregressive mode when True.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not tsfm_output_json.strip():
        return ErrorResult(error="tsfm_output_json is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")
    if task not in ("fit", "inference"):
        return ErrorResult(error="task must be 'fit' or 'inference'")

    try:
        import tsfm_public  # noqa: F401
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    dataset_config = _build_dataset_config(
        timestamp_column,
        target_columns,
        conditional_columns,
        id_columns,
        frequency_sampling or "",
        autoregressive_modeling,
    )

    try:
        with open(tsfm_output_json, "r") as fh:
            tsmodel_pred = json.load(fh)

        output = _TimeSeriesAnomalyDetectionConformalWrapper().run(
            dataset_path,
            dataset_config,
            tsmodel_pred,
            ad_model_checkpoint=ad_model_checkpoint,
            ad_model_save=ad_model_save,
            task=task,
            ad_model_type=ad_model_type,
            n_calibration=n_calibration,
            false_alarm=false_alarm,
        )
    except Exception as exc:
        logger.error("run_tsad failed: %s", exc)
        return ErrorResult(error=str(exc))

    try:
        df = _tsad_output_to_df(output)
        tmp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(tmp_dir, f"tsad_output_{uuid.uuid4()}.csv")
        df.to_csv(csv_path, index=False)
        anomaly_count = (
            int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
        )
    except Exception as exc:
        logger.error("run_tsad result serialisation failed: %s", exc)
        return ErrorResult(error=f"Failed to serialise TSAD output: {exc}")

    return TSADResult(
        status="success",
        results_file=csv_path,
        total_records=len(df),
        anomaly_count=anomaly_count,
        columns=list(df.columns),
        message=(
            f"Anomaly detection complete. {anomaly_count} anomalies in {len(df)} records. "
            f"Results saved to {csv_path}."
        ),
    )


# ── Integrated TSAD (forecasting + anomaly detection in one call) ─────────────


@mcp.tool()
def run_integrated_tsad(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    model_checkpoint: str = "ttm_96_28",
    false_alarm: float = 0.05,
    ad_model_type: str = "timeseries_conformal_adaptive",
    n_calibration: float = 0.2,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: str = "",
    autoregressive_modeling: bool = True,
) -> Union[TSADResult, ErrorResult]:
    """Run end-to-end time-series forecasting + anomaly detection in one call.

    For each target column: runs zero-shot TTM forecasting, then fits a conformal
    AD model and predicts anomaly labels. Saves a combined CSV with anomaly labels
    and KPI scores for all columns.

    Args:
        dataset_path: Path to the dataset (CSV/JSON/XLSX).
        timestamp_column: Name of the timestamp column.
        target_columns: Columns to run forecasting + anomaly detection on.
        model_checkpoint: Pre-trained TTM model name (default: ttm_96_28).
        false_alarm: False alarm rate; default 0.05 → 95% coverage.
        ad_model_type: 'timeseries_conformal' or 'timeseries_conformal_adaptive'.
        n_calibration: Fraction of data for AD calibration (default 0.2).
        conditional_columns: Exogenous feature columns.
        id_columns: ID columns for grouped time series.
        frequency_sampling: Sampling frequency string or '' to auto-detect.
        autoregressive_modeling: Use autoregressive mode when True.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")

    try:
        import tsfm_public  # noqa: F401
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    model_checkpoint = _get_model_checkpoint_path(model_checkpoint)
    dataset_path = _get_dataset_path(dataset_path)

    try:
        ad_model_save = _get_outputs_path("tsad_model_save/")
        os.makedirs(ad_model_save, exist_ok=True)

        with open(model_checkpoint + "/config.json") as _f:
            model_config = json.load(_f)
        df_combined = pd.DataFrame()

        for col in target_columns:
            col_config = _build_dataset_config(
                timestamp_column,
                [col],
                conditional_columns,
                id_columns,
                frequency_sampling,
                autoregressive_modeling,
            )

            # 1. Load and quality-filter data for this column
            data_df = _read_ts_data(dataset_path, dataset_config_dictionary=col_config)
            output_dq = _tsfm_data_quality_filter(
                data_df, col_config, model_config, task="inference"
            )
            data_df_filtered = output_dq["data"]
            col_config_filtered = output_dq["dataset_config_dictionary"]

            if len(data_df_filtered) == 0:
                logger.warning(
                    "Data quality filter removed all data for column %s; skipping.", col
                )
                continue

            # 2. Zero-shot forecasting for this column
            try:
                forecast_output = _get_ttm_hf_inference(
                    data_df_filtered,
                    col_config_filtered,
                    model_config,
                    model_checkpoint,
                )
            except Exception as exc:
                logger.warning("Forecasting failed for column %s: %s", col, exc)
                continue

            inference_data = {
                "target_prediction": forecast_output["target_prediction"].tolist(),
                "timestamp": np.array(forecast_output["timestamp_prediction"])
                .astype(str)
                .tolist(),
                "target_columns": forecast_output["target_columns"],
            }
            # 3. Conformal anomaly detection for this column
            tsmodel_pred = inference_data

            try:
                tsad_output = _TimeSeriesAnomalyDetectionConformalWrapper().run(
                    dataset_path,
                    col_config,
                    tsmodel_pred,
                    ad_model_checkpoint=None,
                    ad_model_save=ad_model_save,
                    task="fit",
                    ad_model_type=ad_model_type,
                    n_calibration=n_calibration,
                    false_alarm=false_alarm,
                )
            except Exception as exc:
                logger.warning("TSAD failed for column %s: %s", col, exc)
                continue

            df_col = _tsad_output_to_df(tsad_output)
            df_combined = pd.concat([df_combined, df_col], axis=0, ignore_index=True)

        if df_combined.empty:
            return ErrorResult(error="No TSAD results produced for any target column.")

        tmp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(tmp_dir, f"integrated_tsad_{uuid.uuid4()}.csv")
        df_combined.to_csv(csv_path, index=False)
        anomaly_count = (
            int(df_combined["anomaly_label"].sum())
            if "anomaly_label" in df_combined.columns
            else 0
        )

    except Exception as exc:
        logger.error("run_integrated_tsad failed: %s", exc)
        return ErrorResult(error=str(exc))

    return TSADResult(
        status="success",
        results_file=csv_path,
        total_records=len(df_combined),
        anomaly_count=anomaly_count,
        columns=list(df_combined.columns),
        message=(
            f"Integrated TSAD complete. {anomaly_count} anomalies in {len(df_combined)} records "
            f"across {len(target_columns)} column(s). Results saved to {csv_path}."
        ),
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
