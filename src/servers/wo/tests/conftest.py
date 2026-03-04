import json
import os

import pytest
import pandas as pd
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

# --- Custom markers ---

requires_wo_data = pytest.mark.skipif(
    not os.path.exists(
        os.environ.get(
            "WO_DATA_DIR",
            os.path.join(
                os.path.dirname(__file__),
                "../../../../tmp/assetopsbench/sample_data",
            ),
        )
    ),
    reason="Work order sample data directory not found (set WO_DATA_DIR)",
)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def reset_data_cache():
    """Reset the module-level data cache between tests."""
    import servers.wo.data as wo_data
    original = dict(wo_data._data)
    wo_data._data = {k: None for k in original}
    yield
    wo_data._data = original


def _make_wo_df() -> pd.DataFrame:
    data = {
        "wo_id": ["WO001", "WO002", "WO003", "WO004"],
        "wo_description": ["Oil Analysis", "Routine Maintenance", "Corrective Repair", "Emergency Fix"],
        "collection": ["compressor", "compressor", "motor", "motor"],
        "primary_code": ["MT010", "MT001", "MT013", "MT013"],
        "primary_code_description": ["Oil Analysis", "Routine Maintenance", "Corrective", "Corrective"],
        "secondary_code": ["MT010b", "MT001a", "MT013a", "MT013b"],
        "secondary_code_description": ["Routine Oil Analysis", "Basic Maint", "Repair", "Emergency"],
        "equipment_id": ["CWC04013", "CWC04013", "CWC04013", "CWC04007"],
        "equipment_name": ["Chiller 13", "Chiller 13", "Chiller 13", "Chiller 7"],
        "preventive": ["TRUE", "TRUE", "FALSE", "FALSE"],
        "work_priority": ["5", "5", "3", "1"],
        "actual_finish": [
            pd.Timestamp("2017-06-01"),
            pd.Timestamp("2017-08-15"),
            pd.Timestamp("2017-11-20"),
            pd.Timestamp("2018-03-10"),
        ],
        "duration": ["3:00", "2:00", "4:00", "6:00"],
        "actual_labor_hours": ["1:00", "1:00", "2:00", "3:00"],
    }
    return pd.DataFrame(data)


def _make_events_df() -> pd.DataFrame:
    data = {
        "event_id": ["E001", "E002", "E003"],
        "event_group": ["WORK_ORDER", "ALERT", "ANOMALY"],
        "event_category": ["PM", "ALERT", "ANOMALY"],
        "event_type": ["MT001", "CR00002", None],
        "description": ["Routine Maintenance", "Temperature Alert", "Anomaly Detected"],
        "equipment_id": ["CWC04013", "CWC04013", "CWC04013"],
        "equipment_name": ["Chiller 13", "Chiller 13", "Chiller 13"],
        "event_time": [
            pd.Timestamp("2017-06-01"),
            pd.Timestamp("2017-07-01"),
            pd.Timestamp("2017-08-01"),
        ],
        "note": [None, "High temp", None],
    }
    return pd.DataFrame(data)


def _make_failure_codes_df() -> pd.DataFrame:
    data = {
        "category": ["Maintenance and Routine Checks", "Maintenance and Routine Checks", "Corrective"],
        "primary_code": ["MT010", "MT001", "MT013"],
        "primary_code_description": ["Oil Analysis", "Routine Maintenance", "Corrective"],
        "secondary_code": ["MT010b", "MT001a", "MT013a"],
        "secondary_code_description": ["Routine Oil Analysis", "Basic Maint", "Repair"],
    }
    return pd.DataFrame(data)


def _make_primary_failure_codes_df() -> pd.DataFrame:
    data = {
        "category": ["Maintenance and Routine Checks", "Maintenance and Routine Checks", "Corrective"],
        "primary_code": ["MT010", "MT001", "MT013"],
        "primary_code_description": ["Oil Analysis", "Routine Maintenance", "Corrective"],
    }
    return pd.DataFrame(data)


def _make_alert_events_df() -> pd.DataFrame:
    data = {
        "equipment_id": ["CWC04013", "CWC04013", "CWC04013"],
        "equipment_name": ["Chiller 13", "Chiller 13", "Chiller 13"],
        "rule": ["CR00002", "CR00002", "CR00002"],
        "start_time": [
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2017-03-01"),
            pd.Timestamp("2017-06-01"),
        ],
        "end_time": [
            pd.Timestamp("2017-01-02"),
            pd.Timestamp("2017-03-02"),
            pd.Timestamp("2017-06-02"),
        ],
        "event_group": ["ALERT", "ALERT", "WORK_ORDER"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_data():
    """Patch all module-level data caches with minimal fixture DataFrames."""
    import servers.wo.data as wo_data

    wo_data._data["wo_events"] = _make_wo_df()
    wo_data._data["events"] = _make_events_df()
    wo_data._data["failure_codes"] = _make_failure_codes_df()
    wo_data._data["primary_failure_codes"] = _make_primary_failure_codes_df()
    wo_data._data["alert_events"] = _make_alert_events_df()
    yield
    wo_data._data = {k: None for k in wo_data._data}


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
