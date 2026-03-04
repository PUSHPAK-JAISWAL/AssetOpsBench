"""Data loading and query helpers for the Work Order MCP server.

All CSVs are loaded lazily on first access.  Missing files log a warning and
return ``None`` so the server starts even when only a subset of data is present.
"""

import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import EventItem, WorkOrderItem

logger = logging.getLogger("wo-mcp-server")

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../tmp/assetopsbench/sample_data")
)
WO_DATA_DIR: str = os.environ.get("WO_DATA_DIR", _DEFAULT_DATA_DIR)


def _csv(filename: str) -> str:
    return os.path.join(WO_DATA_DIR, filename)


# ---------------------------------------------------------------------------
# Lazy cache
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


def load(key: str) -> Optional[pd.DataFrame]:
    """Return the cached DataFrame for *key*, loading it on first call."""
    if _data[key] is not None:
        return _data[key]

    _loaders = {
        "wo_events": _read_wo_events,
        "events": _read_events,
        "alert_events": _read_alert_events,
        "alert_rule": lambda: pd.read_csv(_csv("alert_rule.csv"), dtype=str),
        "alert_rule_fc_mapping": lambda: pd.read_csv(_csv("alert_rule_failure_code_mapping.csv"), dtype=str),
        "anomaly_fc_mapping": lambda: pd.read_csv(_csv("anomaly_to_failure_code_mapping.csv"), dtype=str),
        "failure_codes": lambda: pd.read_csv(_csv("failure_codes.csv"), dtype=str),
        "primary_failure_codes": lambda: pd.read_csv(_csv("primary_failure_codes.csv"), dtype=str),
        "component": lambda: pd.read_csv(_csv("component.csv"), dtype=str),
    }

    try:
        df = _loaders[key]()
        _data[key] = df
        logger.info("Loaded dataset '%s'", key)
        return df
    except FileNotFoundError:
        logger.warning("Data file for '%s' not found in %s", key, WO_DATA_DIR)
        return None
    except Exception as exc:
        logger.error("Failed to load '%s': %s", key, exc)
        return None


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def filter_df(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """Filter *df* by a dict of ``{column: callable_or_query_string}`` conditions."""
    filtered = df.copy()
    for col, cond in conditions.items():
        if callable(cond):
            filtered = filtered[filtered[col].apply(cond)]
        else:
            filtered = filtered.query(f"{col} {cond}")
    if not filtered.empty:
        filtered = filtered.reset_index(drop=True)
    return filtered


def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO date string (YYYY-MM-DD) or raise ValueError."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date must be YYYY-MM-DD, got '{value}'") from exc


def date_conditions(equipment_id: str, date_col: str, start: Optional[str], end: Optional[str]) -> dict:
    """Build a filter-conditions dict for equipment + optional date range."""
    start_dt = parse_date(start)
    end_dt = parse_date(end)
    cond: dict = {
        "equipment_id": lambda x, eid=equipment_id: isinstance(x, str) and x.strip().lower() == eid.strip().lower()
    }
    if start_dt or end_dt:
        cond[date_col] = lambda x, s=start_dt, e=end_dt: (
            (s is None or x >= s) and (e is None or x <= e)
        )
    return cond


def get_transition_matrix(event_df: pd.DataFrame, event_type_col: str) -> pd.DataFrame:
    """Build a row-normalised Markov transition matrix from a sequence of event types."""
    event_types = event_df[event_type_col].tolist()
    counts: dict = defaultdict(lambda: defaultdict(int))
    for cur, nxt in zip(event_types[:-1], event_types[1:]):
        counts[cur][nxt] += 1
    matrix = pd.DataFrame(counts).fillna(0)
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix


# ---------------------------------------------------------------------------
# Row → model converters
# ---------------------------------------------------------------------------


def row_to_wo(row: Any) -> WorkOrderItem:
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


def row_to_event(row: Any) -> EventItem:
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


def fetch_work_orders(
    df: pd.DataFrame,
    equipment_id: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[WorkOrderItem]:
    """Filter *df* by equipment + date range and return ``WorkOrderItem`` list."""
    cond = date_conditions(equipment_id, "actual_finish", start_date, end_date)
    filtered = filter_df(df, cond)
    if filtered is None or filtered.empty:
        return []
    return [row_to_wo(row) for _, row in filtered.iterrows()]
