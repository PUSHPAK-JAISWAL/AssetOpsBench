import json
import os

import pytest
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

# --- Custom markers ---

requires_wo_data = pytest.mark.skipif(
    not os.path.exists(
        os.environ.get(
            "WO_DATA_PATH",
            os.path.join(
                os.path.dirname(__file__),
                "../../../../tmp/assetopsbench/sample_data/all_wo_with_code_component_events.csv",
            ),
        )
    ),
    reason="Work order CSV data not available (set WO_DATA_PATH)",
)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def reset_data_cache():
    """Reset the module-level DataFrame cache between tests."""
    import servers.wo.main as wo_main
    original = wo_main._df
    wo_main._df = None
    yield
    wo_main._df = original


@pytest.fixture
def mock_df(tmp_path):
    """Patch the WO data path with a minimal CSV fixture."""
    import pandas as pd
    import servers.wo.main as wo_main

    csv_path = tmp_path / "wo_test.csv"
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
        "preventive": [True, True, False, False],
        "work_priority": [5, 5, 3, 1],
        "actual_finish": ["2017-06-01", "2017-08-15", "2017-11-20", "2018-03-10"],
        "duration": ["3:00", "2:00", "4:00", "6:00"],
        "actual_labor_hours": ["1:00", "1:00", "2:00", "3:00"],
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)

    with patch.object(wo_main, "WO_DATA_PATH", str(csv_path)):
        wo_main._df = None
        yield
        wo_main._df = None


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
