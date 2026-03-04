import logging
import os
from typing import List, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

# Setup logging — default WARNING so stderr stays quiet when used as MCP server;
# set LOG_LEVEL=INFO (or DEBUG) in the environment to see verbose output.
_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("wo-mcp-server")

WO_DATA_PATH = os.environ.get(
    "WO_DATA_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "../../../src/tmp/assetopsbench/sample_data/all_wo_with_code_component_events.csv",
    ),
)

mcp = FastMCP("WorkOrderAgent")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_df: Optional[pd.DataFrame] = None


def _load_data() -> Optional[pd.DataFrame]:
    global _df
    if _df is not None:
        return _df
    try:
        df = pd.read_csv(WO_DATA_PATH)
        df["actual_finish"] = pd.to_datetime(df["actual_finish"], errors="coerce")
        _df = df
        logger.info(f"Loaded {len(df)} work order records from {WO_DATA_PATH}")
        return _df
    except Exception as e:
        logger.error(f"Failed to load work order data: {e}")
        return None


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ErrorResult(BaseModel):
    error: str


class EquipmentListResult(BaseModel):
    total_equipment: int
    equipment_ids: List[str]


class WorkOrder(BaseModel):
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
    total_work_orders: int
    work_orders: List[WorkOrder]
    message: str


class WorkOrderSummaryResult(BaseModel):
    equipment_id: str
    start_date: Optional[str]
    end_date: Optional[str]
    total_work_orders: int
    preventive_count: int
    corrective_count: int
    by_primary_code: dict
    by_collection: dict
    message: str


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_equipment() -> Union[EquipmentListResult, ErrorResult]:
    """Returns a list of all equipment IDs available in the work order dataset."""
    df = _load_data()
    if df is None:
        return ErrorResult(error="Work order data not available")
    ids = sorted(df["equipment_id"].dropna().unique().tolist())
    return EquipmentListResult(total_equipment=len(ids), equipment_ids=ids)


@mcp.tool()
def get_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrdersResult, ErrorResult]:
    """Retrieves work orders for a specific equipment within an optional date range.

    Args:
        equipment_id: The equipment identifier (e.g. "CWC04013").
        start_date: Optional ISO date string for the start of the range (inclusive).
        end_date: Optional ISO date string for the end of the range (exclusive).
    """
    df = _load_data()
    if df is None:
        return ErrorResult(error="Work order data not available")

    filtered = df[df["equipment_id"] == equipment_id]
    if filtered.empty:
        return ErrorResult(error=f"No work orders found for equipment_id '{equipment_id}'")

    if start_date:
        try:
            filtered = filtered[filtered["actual_finish"] >= pd.to_datetime(start_date)]
        except Exception as e:
            return ErrorResult(error=f"Invalid start_date: {e}")

    if end_date:
        try:
            filtered = filtered[filtered["actual_finish"] < pd.to_datetime(end_date)]
        except Exception as e:
            return ErrorResult(error=f"Invalid end_date: {e}")

    records: List[WorkOrder] = []
    for _, row in filtered.iterrows():
        records.append(
            WorkOrder(
                wo_id=str(row.get("wo_id", "")),
                wo_description=str(row.get("wo_description", "")),
                collection=str(row.get("collection", "")),
                primary_code=str(row.get("primary_code", "")),
                primary_code_description=str(row.get("primary_code_description", "")),
                secondary_code=str(row.get("secondary_code", "")),
                secondary_code_description=str(row.get("secondary_code_description", "")),
                equipment_id=str(row.get("equipment_id", "")),
                equipment_name=str(row.get("equipment_name", "")),
                preventive=bool(row.get("preventive", False)),
                work_priority=int(row["work_priority"]) if pd.notna(row.get("work_priority")) else None,
                actual_finish=row["actual_finish"].isoformat() if pd.notna(row.get("actual_finish")) else None,
                duration=str(row.get("duration", "")) if pd.notna(row.get("duration")) else None,
                actual_labor_hours=str(row.get("actual_labor_hours", "")) if pd.notna(row.get("actual_labor_hours")) else None,
            )
        )

    return WorkOrdersResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total_work_orders=len(records),
        work_orders=records,
        message=f"Found {len(records)} work orders for equipment '{equipment_id}'.",
    )


@mcp.tool()
def summarize_work_orders(
    equipment_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Union[WorkOrderSummaryResult, ErrorResult]:
    """Summarizes work orders for a specific equipment within an optional date range.

    Returns counts broken down by type (primary_code) and collection, plus
    preventive vs. corrective totals.

    Args:
        equipment_id: The equipment identifier (e.g. "CWC04013").
        start_date: Optional ISO date string for the start of the range (inclusive).
        end_date: Optional ISO date string for the end of the range (exclusive).
    """
    df = _load_data()
    if df is None:
        return ErrorResult(error="Work order data not available")

    filtered = df[df["equipment_id"] == equipment_id]
    if filtered.empty:
        return ErrorResult(error=f"No work orders found for equipment_id '{equipment_id}'")

    if start_date:
        try:
            filtered = filtered[filtered["actual_finish"] >= pd.to_datetime(start_date)]
        except Exception as e:
            return ErrorResult(error=f"Invalid start_date: {e}")

    if end_date:
        try:
            filtered = filtered[filtered["actual_finish"] < pd.to_datetime(end_date)]
        except Exception as e:
            return ErrorResult(error=f"Invalid end_date: {e}")

    total = len(filtered)
    preventive_count = int((filtered["preventive"] == True).sum())  # noqa: E712
    corrective_count = total - preventive_count

    by_primary_code: dict = (
        filtered.groupby("primary_code").size().to_dict()
        if total > 0
        else {}
    )
    by_collection: dict = (
        filtered.groupby("collection").size().to_dict()
        if total > 0
        else {}
    )

    return WorkOrderSummaryResult(
        equipment_id=equipment_id,
        start_date=start_date,
        end_date=end_date,
        total_work_orders=total,
        preventive_count=preventive_count,
        corrective_count=corrective_count,
        by_primary_code=by_primary_code,
        by_collection=by_collection,
        message=f"Summarized {total} work orders for equipment '{equipment_id}'.",
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
