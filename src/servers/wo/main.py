"""Work Order MCP server.

Exposes work-order and event data as FastMCP tools.  All data is loaded from a
configurable directory (env var ``WO_DATA_DIR``).  Every data file is read
lazily on first use so the server starts even when only a subset of files is
available.
"""

import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("wo-mcp-server")

# ---------------------------------------------------------------------------
# Data directory — default to the bundled sample data shipped with the repo
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "../../tmp/assetopsbench/sample_data",
    )
)
WO_DATA_DIR = os.environ.get("WO_DATA_DIR", _DEFAULT_DATA_DIR)


def _csv(filename: str) -> str:
    return os.path.join(WO_DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Lazy data loading — each DataFrame is None until first access
# ---------------------------------------------------------------------------

_data: Dict[str, Optional[pd.DataFrame]] = {
    "wo_events": None,
    "events": None,
    "alert_events": None,
    "alert_rule": None,
    "alert_rule_fc_mapping": None,
    "anomaly_fc_mapping": None,
    "failure_codes": None,
    "primary_failure_codes": None,
    "component": None,
}


def _load(key: str) -> Optional[pd.DataFrame]:
    if _data[key] is not None:
        return _data[key]

    loaders = {
        "wo_events": lambda: _read_wo_events(),
        "events": lambda: _read_events(),
        "alert_events": lambda: _read_alert_events(),
        "alert_rule": lambda: pd.read_csv(_csv("alert_rule.csv"), dtype=str),
        "alert_rule_fc_mapping": lambda: pd.read_csv(_csv("alert_rule_failure_code_mapping.csv"), dtype=str),
        "anomaly_fc_mapping": lambda: pd.read_csv(_csv("anomaly_to_failure_code_mapping.csv"), dtype=str),
        "failure_codes": lambda: pd.read_csv(_csv("failure_codes.csv"), dtype=str),
        "primary_failure_codes": lambda: pd.read_csv(_csv("primary_failure_codes.csv"), dtype=str),
        "component": lambda: pd.read_csv(_csv("component.csv"), dtype=str),
    }

    try:
        df = loaders[key]()
        _data[key] = df
        logger.info("Loaded dataset '%s'", key)
        return df
    except FileNotFoundError:
        logger.warning("Data file for '%s' not found in %s", key, WO_DATA_DIR)
        return None
    except Exception as exc:
        logger.error("Failed to load '%s': %s", key, exc)
        return None


def _read_wo_events() -> pd.DataFrame:
    df = pd.read_csv(_csv("all_wo_with_code_component_events.csv"), dtype=str)
    df["actual_finish"] = pd.to_datetime(df["actual_finish"], format="%m/%d/%y %H:%M", errors="coerce")
    return df


def _read_events() -> pd.DataFrame:
    df = pd.read_csv(_csv("event.csv"), dtype=str)
    df["event_time"] = pd.to_datetime(df["event_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return df


def _read_alert_events() -> pd.DataFrame:
    df = pd.read_csv(_csv("alert_events.csv"), dtype=str)
    df["start_time"] = pd.to_datetime(df["start_time"], format="%m/%d/%y %H:%M", errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], format="%m/%d/%y %H:%M", errors="coerce")
    return df


def _alert_to_wo_df() -> Optional[pd.DataFrame]:
    """Joined alert-rule → failure-code lookup table."""
    fc_mapping = _load("alert_rule_fc_mapping")
    pfc = _load("primary_failure_codes")
    if fc_mapping is None or pfc is None:
        return None
    try:
        merged = pd.merge(
            fc_mapping,
            pfc[["category", "primary_code", "primary_code_description"]],
            on="primary_code",
            suffixes=("_rule", ""),
        ).drop(columns=["primary_code_description_rule"], errors="ignore")
        return merged
    except Exception as exc:
        logger.error("Failed to build alert_to_wo join: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_df(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    filtered = df.copy()
    for col, cond in conditions.items():
        if callable(cond):
            filtered = filtered[filtered[col].apply(cond)]
        else:
            filtered = filtered.query(f"{col} {cond}")
    if filtered is not None and not filtered.empty:
        filtered = filtered.reset_index(drop=True)
    return filtered


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got '{value}'") from exc


def _date_conditions(equipment_id: str, date_col: str, start: Optional[str], end: Optional[str]) -> dict:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower()
    }
    if start_dt or end_dt:
        cond[date_col] = lambda x, s=start_dt, e=end_dt: (
            (s is None or x >= s) and (e is None or x <= e)
        )
    return cond


def _get_transition_matrix(event_df: pd.DataFrame, event_type_col: str) -> pd.DataFrame:
    event_types = event_df[event_type_col].tolist()
    counts: dict = defaultdict(lambda: defaultdict(int))
    for cur, nxt in zip(event_types[:-1], event_types[1:]):
        counts[cur][nxt] += 1
    matrix = pd.DataFrame(counts).fillna(0)
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ErrorResult(BaseModel):
    error: str


class WorkOrderItem(BaseModel):
    wo_id: str
    wo_description: str
    collection: str
    primary_code: str
    primary_code_description: str
    secondary_code: str
    secondary_code_description: str
    equipment_id: str
    equipment_name: str
    preventive: bool
    work_priority: Optional[int]
    actual_finish: Optional[str]
    duration: Optional[str]
    actual_labor_hours: Optional[str]


class WorkOrdersResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total: int
    work_orders: List[WorkOrderItem]
    message: str


class EventItem(BaseModel):
    event_id: str
    event_group: str
    event_category: str
    event_type: Optional[str]
    description: Optional[str]
    equipment_id: str
    equipment_name: str
    event_time: str
    note: Optional[str]


class EventsResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total: int
    events: List[EventItem]
    message: str


class FailureCodeItem(BaseModel):
    category: str
    primary_code: str
    primary_code_description: str
    secondary_code: str
    secondary_code_description: str


class FailureCodesResult(BaseModel):
    total: int
    failure_codes: List[FailureCodeItem]


class WorkOrderDistributionEntry(BaseModel):
    category: str
    primary_code: str
    primary_code_description: str
    secondary_code: str
    secondary_code_description: str
    count: int


class WorkOrderDistributionResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_work_orders: int
    distribution: List[WorkOrderDistributionEntry]
    message: str


class NextWorkOrderEntry(BaseModel):
    category: str
    primary_code: str
    primary_code_description: str
    probability: float


class NextWorkOrderPredictionResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    last_work_order_type: str
    predictions: List[NextWorkOrderEntry]
    message: str


class AlertToFailureEntry(BaseModel):
    transition: str
    probability: float
    average_hours_to_maintenance: Optional[float]


class AlertToFailureResult(BaseModel):
    equipment_id: str
    rule_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_alerts_analyzed: int
    transitions: List[AlertToFailureEntry]
    message: str


# ---------------------------------------------------------------------------
# Row → model helpers
# ---------------------------------------------------------------------------


def _row_to_wo(row: Any) -> WorkOrderItem:
    return WorkOrderItem(
        wo_id=str(row.get("wo_id", "")),
        wo_description=str(row.get("wo_description", "")),
        collection=str(row.get("collection", "")),
        primary_code=str(row.get("primary_code", "")),
        primary_code_description=str(row.get("primary_code_description", "")),
        secondary_code=str(row.get("secondary_code", "")),
        secondary_code_description=str(row.get("secondary_code_description", "")),
        equipment_id=str(row.get("equipment_id", "")),
        equipment_name=str(row.get("equipment_name", "")),
        preventive=str(row.get("preventive", "")).upper() == "TRUE",
        work_priority=int(row["work_priority"]) if pd.notna(row.get("work_priority")) else None,
        actual_finish=row["actual_finish"].isoformat() if pd.notna(row.get("actual_finish")) else None,
        duration=str(row.get("duration", "")) if pd.notna(row.get("duration")) else None,
        actual_labor_hours=str(row.get("actual_labor_hours", "")) if pd.notna(row.get("actual_labor_hours")) else None,
    )


def _row_to_event(row: Any) -> EventItem:
    return EventItem(
        event_id=str(row.get("event_id", "")),
        event_group=str(row.get("event_group", "")),
        event_category=str(row.get("event_category", "")),
        event_type=str(row["event_type"]) if pd.notna(row.get("event_type")) else None,
        description=str(row["description"]) if pd.notna(row.get("description")) else None,
        equipment_id=str(row.get("equipment_id", "")),
        equipment_name=str(row.get("equipment_name", "")),
        event_time=row["event_time"].isoformat() if pd.notna(row.get("event_time")) else "",
        note=str(row["note"]) if pd.notna(row.get("note")) else None,
    )


def _fetch_work_orders(
    df: pd.DataFrame,
    equipment_id: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[WorkOrderItem]:
    cond = _date_conditions(equipment_id, "actual_finish", start_date, end_date)
    filtered = _filter_df(df, cond)
    if filtered is None or filtered.empty:
        return []
    return [_row_to_wo(row) for _, row in filtered.iterrows()]


# ---------------------------------------------------------------------------
# MCP server + tools
# ---------------------------------------------------------------------------

mcp = FastMCP("WorkOrderAgent")


@mcp.tool()
def get_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieve all work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = _load("wo_events")
    if df is None:
        return ErrorResult(error="Work order data not available")
    try:
        wos = _fetch_work_orders(df, equipment_id, start_date, end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))
    if not wos:
        return ErrorResult(error=f"No work orders found for equipment_id '{equipment_id}'")
    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(wos),
        work_orders=wos,
        message=f"Found {len(wos)} work orders for '{equipment_id}'.",
    )


@mcp.tool()
def get_preventive_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieve only preventive work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = _load("wo_events")
    if df is None:
        return ErrorResult(error="Work order data not available")
    try:
        wos = _fetch_work_orders(df[df["preventive"] == "TRUE"], equipment_id, start_date, end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))
    if not wos:
        return ErrorResult(error=f"No preventive work orders found for equipment_id '{equipment_id}'")
    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(wos),
        work_orders=wos,
        message=f"Found {len(wos)} preventive work orders for '{equipment_id}'.",
    )


@mcp.tool()
def get_corrective_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieve only corrective work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = _load("wo_events")
    if df is None:
        return ErrorResult(error="Work order data not available")
    try:
        wos = _fetch_work_orders(df[df["preventive"] == "FALSE"], equipment_id, start_date, end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))
    if not wos:
        return ErrorResult(error=f"No corrective work orders found for equipment_id '{equipment_id}'")
    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(wos),
        work_orders=wos,
        message=f"Found {len(wos)} corrective work orders for '{equipment_id}'.",
    )


@mcp.tool()
def get_events(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[EventsResult, ErrorResult]:
    """Retrieve all events (work orders, alerts, anomalies) for a specific equipment within an optional date range.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    df = _load("events")
    if df is None:
        return ErrorResult(error="Event data not available")
    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower()
    }
    if start_dt or end_dt:
        cond["event_time"] = lambda x, s=start_dt, e=end_dt: (
            (s is None or x >= s) and (e is None or x <= e)
        )

    filtered = _filter_df(df, cond)
    if filtered is None or filtered.empty:
        return ErrorResult(error=f"No events found for equipment_id '{equipment_id}'")

    events = [_row_to_event(row) for _, row in filtered.iterrows()]
    return EventsResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total=len(events),
        events=events,
        message=f"Found {len(events)} events for '{equipment_id}'.",
    )


@mcp.tool()
def get_failure_codes() -> Union[FailureCodesResult, ErrorResult]:
    """Retrieve all available failure codes with their categories and descriptions."""
    df = _load("failure_codes")
    if df is None:
        return ErrorResult(error="Failure codes data not available")

    items = [
        FailureCodeItem(
            category=str(row.get("category", "")),
            primary_code=str(row.get("primary_code", "")),
            primary_code_description=str(row.get("primary_code_description", "")),
            secondary_code=str(row.get("secondary_code", "")),
            secondary_code_description=str(row.get("secondary_code_description", "")),
        )
        for _, row in df.iterrows()
    ]
    return FailureCodesResult(total=len(items), failure_codes=items)


@mcp.tool()
def get_work_order_distribution(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrderDistributionResult, ErrorResult]:
    """Calculate the distribution of work order types (by failure code) for a specific equipment.

    Returns counts per (primary_code, secondary_code) pair, sorted by frequency descending.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    wo_df = _load("wo_events")
    fc_df = _load("failure_codes")
    if wo_df is None:
        return ErrorResult(error="Work order data not available")
    if fc_df is None:
        return ErrorResult(error="Failure codes data not available")

    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    filtered = wo_df[wo_df["equipment_id"] == equipment_id].copy()
    if start_dt:
        filtered = filtered[filtered["actual_finish"] >= start_dt]
    if end_dt:
        filtered = filtered[filtered["actual_finish"] <= end_dt]

    if filtered.empty:
        return ErrorResult(error=f"No work orders found for equipment_id '{equipment_id}'")

    counts = (
        filtered.groupby(["primary_code", "secondary_code"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    distribution: List[WorkOrderDistributionEntry] = []
    for _, row in counts.iterrows():
        match = fc_df[
            (fc_df["primary_code"] == row["primary_code"])
            & (fc_df["secondary_code"] == row["secondary_code"])
        ]
        if match.empty:
            continue
        m = match.iloc[0]
        distribution.append(
            WorkOrderDistributionEntry(
                category=str(m.get("category", "")),
                primary_code=str(m.get("primary_code", "")),
                primary_code_description=str(m.get("primary_code_description", "")),
                secondary_code=str(m.get("secondary_code", "")),
                secondary_code_description=str(m.get("secondary_code_description", "")),
                count=int(row["count"]),
            )
        )

    return WorkOrderDistributionResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total_work_orders=int(filtered.shape[0]),
        distribution=distribution,
        message=f"Distribution across {len(distribution)} failure code(s) for '{equipment_id}'.",
    )


@mcp.tool()
def predict_next_work_order(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[NextWorkOrderPredictionResult, ErrorResult]:
    """Predict the probabilities of the next expected work order types based on historical transition patterns.

    Uses a Markov-chain transition matrix built from the sequence of past work order
    primary codes to estimate what type of work order is likely to follow the most
    recent one.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    wo_df = _load("wo_events")
    pfc_df = _load("primary_failure_codes")
    if wo_df is None:
        return ErrorResult(error="Work order data not available")

    try:
        start_dt = _parse_date(start_date)
        end_dt = _parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    cond = _date_conditions(equipment_id, "actual_finish", start_date, end_date)
    filtered = _filter_df(wo_df, cond)
    if filtered is None or filtered.empty:
        return ErrorResult(error=f"No historical work orders found for equipment_id '{equipment_id}'")

    filtered = filtered.sort_values("actual_finish").reset_index(drop=True)
    transition_matrix = _get_transition_matrix(filtered, "primary_code")
    last_type = filtered.iloc[-1]["primary_code"]

    if last_type not in transition_matrix.index:
        return ErrorResult(error=f"No transition data for last work order type '{last_type}'")

    raw_predictions = sorted(
        transition_matrix.loc[last_type].items(),
        key=lambda t: t[1],
        reverse=True,
    )

    predictions: List[NextWorkOrderEntry] = []
    for primary_code, prob in raw_predictions:
        if pfc_df is not None:
            match = pfc_df[pfc_df["primary_code"] == primary_code]
            if not match.empty:
                m = match.iloc[0]
                predictions.append(
                    NextWorkOrderEntry(
                        category=str(m.get("category", "")),
                        primary_code=primary_code,
                        primary_code_description=str(m.get("primary_code_description", "")),
                        probability=float(prob),
                    )
                )
                continue
        predictions.append(
            NextWorkOrderEntry(category="", primary_code=primary_code, primary_code_description="", probability=float(prob))
        )

    return NextWorkOrderPredictionResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        last_work_order_type=last_type,
        predictions=predictions,
        message=f"Predicted next work order for '{equipment_id}' based on last type '{last_type}'.",
    )


@mcp.tool()
def analyze_alert_to_failure(
    equipment_id: str,
    rule_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[AlertToFailureResult, ErrorResult]:
    """Analyze the relationship between a specific alert rule and subsequent maintenance events for an equipment.

    Computes the probability that each alert occurrence leads to a work order (vs no
    maintenance) and the average time-to-maintenance in hours.

    Args:
        equipment_id: Equipment identifier, e.g. ``"CWC04013"``.
        rule_id: Alert rule identifier, e.g. ``"CR00002"``.
        start_date: Start of date range (inclusive), format ``YYYY-MM-DD``.
        end_date: End of date range (inclusive), format ``YYYY-MM-DD``.
    """
    alert_df = _load("alert_events")
    if alert_df is None:
        return ErrorResult(error="Alert events data not available")

    try:
        _parse_date(start_date)
        _parse_date(end_date)
    except ValueError as exc:
        return ErrorResult(error=str(exc))

    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower(),
        "rule": lambda x, rid=rule_id: isinstance(x, str) and x.strip().lower() == rid.strip().lower(),
    }
    filtered = _filter_df(alert_df, cond)
    if filtered is None or filtered.empty:
        return ErrorResult(error=f"No alert events found for equipment '{equipment_id}' and rule '{rule_id}'")

    filtered = filtered.sort_values("start_time").reset_index(drop=True)

    # Compute transitions from alert occurrences to maintenance
    transitions: List[str] = []
    time_diffs: List[float] = []
    for i in range(len(filtered) - 1):
        if str(filtered.iloc[i].get("rule", "")).strip().lower() == rule_id.strip().lower():
            for j in range(i + 1, len(filtered)):
                if str(filtered.iloc[j].get("event_group", "")).upper() == "WORK_ORDER":
                    transitions.append("WORK_ORDER")
                    diff = filtered.iloc[j]["start_time"] - filtered.iloc[i]["start_time"]
                    time_diffs.append(diff.total_seconds() / 3600)
                    break
            else:
                transitions.append("No Maintenance")

    if not transitions:
        return ErrorResult(error="Insufficient alert history to compute transitions")

    counts = Counter(transitions)
    total = len(transitions)
    wo_times = time_diffs if time_diffs else []

    entries: List[AlertToFailureEntry] = []
    for transition, count in sorted(counts.items(), key=lambda t: t[1], reverse=True):
        avg_hours = sum(wo_times) / len(wo_times) if transition == "WORK_ORDER" and wo_times else None
        entries.append(
            AlertToFailureEntry(
                transition=transition,
                probability=count / total,
                average_hours_to_maintenance=avg_hours,
            )
        )

    return AlertToFailureResult(
        equipment_id=equipment_id,
        rule_id=rule_id,
        start_date=start_date,
        end_date=end_date,
        total_alerts_analyzed=total,
        transitions=entries,
        message=f"Analyzed {total} alert occurrences for rule '{rule_id}' on '{equipment_id}'.",
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
