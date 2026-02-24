"""TSFM (Time Series Foundation Model) MCP Server.

Exposes tools for time-series forecasting, finetuning, and anomaly detection
using the TinyTimeMixer (TTM) foundation models from src/tsfmagent/.

Tools:
  get_ai_tasks          – list available AI task types (static, no deps)
  get_tsfm_models       – list available pre-trained model checkpoints (static)
  run_tsfm_forecasting  – zero-shot TTM inference on a dataset
  run_tsfm_finetuning   – few-shot finetuning of a TTM model
  run_tsad              – conformal anomaly detection on TSFM forecasts
  run_integrated_tsad   – end-to-end: forecasting + anomaly detection

Heavy ML dependencies (tsfm_public, transformers, torch) are imported lazily;
the server starts and exposes the static tools even when they are absent.

Required environment variables (path resolution via tsfmagent/utils/utils.py):
  PATH_TO_MODELS_DIR    – directory containing TTM model checkpoint folders
  PATH_TO_DATASETS_DIR  – base directory for resolving relative dataset paths
  PATH_TO_OUTPUTS_DIR   – base directory for resolving output/save paths
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")


# ── Bootstrap: add src/ so that tsfmagent is importable ───────────────────────
# File lives at mcp/servers/tsfm/main.py  →  parents[3] is the repo root.

_SRC = Path(__file__).resolve().parents[3] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Static data ───────────────────────────────────────────────────────────────

_AI_TASKS = [
    {"task_id": "tsfm_integrated_tsad",
     "task_description": "Time series Anomaly detection"},
    {"task_id": "tsfm_forecasting",
     "task_description": "Time series Multivariate Forecasting"},
    {"task_id": "tsfm_forecasting_finetune",
     "task_description": "Finetuning of Multivariate Forecasting models"},
    {"task_id": "tsfm_forecasting_evaluation",
     "task_description": "Evaluation of Forecasting models"},
]

_TSFM_MODELS = [
    {"model_id": "ttm_96_28", "model_checkpoint": "ttm_96_28",
     "model_description": "Pretrained forecasting model with context length 96"},
    {"model_id": "ttm_512_96", "model_checkpoint": "ttm_512_96",
     "model_description": "Pretrained forecasting model with context length 512"},
    {"model_id": "ttm_energy_96_28", "model_checkpoint": "ttm_96_28",
     "model_description": "Pretrained forecasting model tuned on energy data with context length 96"},
    {"model_id": "ttm_energy_512_96", "model_checkpoint": "ttm_512_96",
     "model_description": "Pretrained forecasting model tuned on energy data with context length 512"},
]


# ── Result models ─────────────────────────────────────────────────────────────

class ErrorResult(BaseModel):
    error: str


class AITaskEntry(BaseModel):
    task_id: str
    task_description: str


class AITasksResult(BaseModel):
    tasks: List[AITaskEntry]


class TSFMModelEntry(BaseModel):
    model_id: str
    model_checkpoint: str
    model_description: str


class TSFMModelsResult(BaseModel):
    models: List[TSFMModelEntry]


class ForecastingResult(BaseModel):
    status: str
    results_file: str
    dataquality_summary: Optional[Any] = None
    message: str


class FinetuningResult(BaseModel):
    status: str
    model_checkpoint: str
    results_file: str
    message: str


class TSADResult(BaseModel):
    status: str
    results_file: str
    total_records: int
    anomaly_count: int
    columns: List[str]
    message: str


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_dataset_config(
    timestamp_column: str,
    target_columns: List[str],
    conditional_columns: Optional[List[str]],
    id_columns: Optional[List[str]],
    frequency_sampling: str,
    autoregressive_modeling: bool,
) -> dict:
    """Build the dataset_config_dictionary expected by TSFMForecastWrapper."""
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


def _tsad_output_to_df(output: dict):
    """Convert a TSAD wrapper output dict to a pandas DataFrame.

    The 'KPI' key is stored at index [0] by the wrapper; all other values
    are array-like. Returns a DataFrame ready to be saved as CSV.
    """
    import numpy as np
    import pandas as pd

    kpi = output.pop("KPI", None)
    df = pd.DataFrame.from_dict({k: np.array(v) for k, v in output.items()})
    if kpi is not None:
        df["KPI"] = kpi[0] if (hasattr(kpi, "__len__") and not isinstance(kpi, str)) else kpi
    return df


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("TSFMAgent")


# ── Static tools (no external dependencies) ───────────────────────────────────

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
        from tsfmagent.tools.tsfm.TSFMWrapper import TSFMForecastWrapper
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    dataset_config = _build_dataset_config(
        timestamp_column, target_columns, conditional_columns,
        id_columns, frequency_sampling, autoregressive_modeling,
    )

    try:
        result = TSFMForecastWrapper().run(
            dataset_path,
            dataset_config,
            model_checkpoint,
            model_type="ttm",
            model_source="modelcatalog",
            task="inference",
            forecast_horizon=forecast_horizon,
            include_dataquality_summary=include_dataquality_summary,
        )
    except Exception as exc:
        logger.error("run_tsfm_forecasting failed: %s", exc)
        return ErrorResult(error=str(exc))

    if result.get("error_message"):
        return ErrorResult(error=result["error_message"])

    results_file = result.get("results_file", "")
    return ForecastingResult(
        status="success",
        results_file=results_file,
        dataquality_summary=result.get("dataquality_summary"),
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
            Default 0.05 (5 %) as recommended by the TSFM procedure.
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
        from tsfmagent.tools.tsfm.TSFMWrapper import TSFMForecastWrapper
        from tsfmagent.tools.tsfm.tsfm_hf import find_largest_tsfm_checkpoint_directory
        from tsfmagent.utils.utils import get_outputs_path
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    dataset_config = _build_dataset_config(
        timestamp_column, target_columns, conditional_columns,
        id_columns, frequency_sampling, autoregressive_modeling,
    )

    try:
        result = TSFMForecastWrapper().run(
            dataset_path,
            dataset_config,
            model_checkpoint,
            model_type="ttm",
            model_source="modelcatalog",
            task="finetuning",
            forecast_horizon=forecast_horizon,
            include_dataquality_summary=include_dataquality_summary,
            n_finetune=n_finetune,
            n_calibration=n_calibration,
            n_test=n_test,
            save_model_dir=save_model_dir,
        )
    except Exception as exc:
        logger.error("run_tsfm_finetuning failed: %s", exc)
        return ErrorResult(error=str(exc))

    if result.get("error_message"):
        return ErrorResult(error=result["error_message"])

    # Resolve the actual HF Trainer checkpoint directory written as checkpoint-N/.
    try:
        abs_save_dir = get_outputs_path(save_model_dir)
        saved_checkpoint = (
            find_largest_tsfm_checkpoint_directory(abs_save_dir + "/fewshot/") + "/"
        )
    except Exception as exc:
        logger.warning("Could not resolve finetuned checkpoint path: %s", exc)
        saved_checkpoint = save_model_dir

    results_file = result.get("results_file", "")
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

    tsfm_output_json must be the results_file path returned by
    run_tsfm_forecasting. The tool fits (or loads) a conformal AD model and
    saves anomaly-labelled predictions to a CSV file.

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
        import json
        from tsfmagent.tools.tsad.TimeSeriesAnomalyDetectionWrapper import (
            TimeSeriesAnomalyDetectionConformalWrapper,
        )
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    dataset_config = _build_dataset_config(
        timestamp_column, target_columns, conditional_columns,
        id_columns, frequency_sampling or "", autoregressive_modeling,
    )

    try:
        with open(tsfm_output_json, "r") as fh:
            tsmodel_pred = json.load(fh)

        output = TimeSeriesAnomalyDetectionConformalWrapper().run(
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
        anomaly_count = int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
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

    For each target column: runs zero-shot TTM forecasting, then fits a
    conformal AD model and predicts anomaly labels. Saves a single combined
    CSV with anomaly labels and KPI scores for all columns.

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
        import json
        import pandas as pd
        from tsfmagent.tools.tsfm.TSFMWrapper import TSFMForecastWrapper
        from tsfmagent.tools.tsad.TimeSeriesAnomalyDetectionWrapper import (
            TimeSeriesAnomalyDetectionConformalWrapper,
        )
        from tsfmagent.utils.utils import get_outputs_path
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    try:
        ad_model_save = get_outputs_path("tsad_model_save/")
        os.makedirs(ad_model_save, exist_ok=True)

        df_combined = pd.DataFrame()
        for col in target_columns:
            col_config = _build_dataset_config(
                timestamp_column, [col], conditional_columns,
                id_columns, frequency_sampling, autoregressive_modeling,
            )

            # 1. Zero-shot forecasting for this column
            forecast_result = TSFMForecastWrapper().run(
                dataset_path,
                col_config,
                model_checkpoint,
                model_type="ttm",
                model_source="modelcatalog",
                task="inference",
            )
            if forecast_result.get("error_message"):
                logger.warning(
                    "Forecasting failed for column %s: %s",
                    col, forecast_result["error_message"],
                )
                continue

            results_file = forecast_result.get("results_file", "")
            if not results_file:
                logger.warning("No results_file for column %s; skipping.", col)
                continue

            # 2. Conformal anomaly detection for this column
            with open(results_file, "r") as fh:
                tsmodel_pred = json.load(fh)

            tsad_output = TimeSeriesAnomalyDetectionConformalWrapper().run(
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
