"""Tests for Work Order MCP server tools.

Unit tests use a mocked CSV fixture; integration tests require the real dataset
(skipped unless WO_DATA_PATH points to a valid CSV).
"""

import pytest
from servers.wo.main import mcp
from .conftest import requires_wo_data, call_tool


# ---------------------------------------------------------------------------
# list_equipment
# ---------------------------------------------------------------------------


class TestListEquipment:
    @pytest.mark.anyio
    async def test_returns_equipment_list(self, mock_df):
        data = await call_tool(mcp, "list_equipment", {})
        assert "equipment_ids" in data
        assert "CWC04013" in data["equipment_ids"]
        assert "CWC04007" in data["equipment_ids"]
        assert data["total_equipment"] == 2

    @requires_wo_data
    @pytest.mark.anyio
    async def test_integration_returns_equipment(self):
        data = await call_tool(mcp, "list_equipment", {})
        assert "equipment_ids" in data
        assert data["total_equipment"] > 0
        assert any("CWC" in eq for eq in data["equipment_ids"])


# ---------------------------------------------------------------------------
# get_work_orders
# ---------------------------------------------------------------------------


class TestGetWorkOrders:
    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_df):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_returns_all_records(self, mock_df):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "CWC04013"})
        assert data["total_work_orders"] == 3
        assert len(data["work_orders"]) == 3

    @pytest.mark.anyio
    async def test_date_range_filter(self, mock_df):
        data = await call_tool(
            mcp,
            "get_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2018-01-01"},
        )
        assert data["total_work_orders"] == 3
        for wo in data["work_orders"]:
            assert wo["actual_finish"] is not None
            assert "2017" in wo["actual_finish"]

    @pytest.mark.anyio
    async def test_invalid_start_date(self, mock_df):
        data = await call_tool(
            mcp, "get_work_orders", {"equipment_id": "CWC04013", "start_date": "not-a-date"}
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_work_order_fields(self, mock_df):
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "CWC04013"})
        wo = data["work_orders"][0]
        assert "wo_id" in wo
        assert "wo_description" in wo
        assert "primary_code" in wo
        assert "preventive" in wo
        assert "equipment_id" in wo

    @pytest.mark.anyio
    async def test_preventive_filter_via_date(self, mock_df):
        # CWC04013 has 2 preventive and 1 corrective; all in 2017
        data = await call_tool(mcp, "get_work_orders", {"equipment_id": "CWC04013"})
        preventive = [wo for wo in data["work_orders"] if wo["preventive"]]
        corrective = [wo for wo in data["work_orders"] if not wo["preventive"]]
        assert len(preventive) == 2
        assert len(corrective) == 1

    @requires_wo_data
    @pytest.mark.anyio
    async def test_integration_cwc04013_2017(self):
        data = await call_tool(
            mcp,
            "get_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2018-01-01"},
        )
        assert "work_orders" in data
        assert data["total_work_orders"] > 0


# ---------------------------------------------------------------------------
# summarize_work_orders
# ---------------------------------------------------------------------------


class TestSummarizeWorkOrders:
    @pytest.mark.anyio
    async def test_unknown_equipment(self, mock_df):
        data = await call_tool(mcp, "summarize_work_orders", {"equipment_id": "UNKNOWN"})
        assert "error" in data

    @pytest.mark.anyio
    async def test_summary_counts(self, mock_df):
        data = await call_tool(mcp, "summarize_work_orders", {"equipment_id": "CWC04013"})
        assert data["total_work_orders"] == 3
        assert data["preventive_count"] == 2
        assert data["corrective_count"] == 1

    @pytest.mark.anyio
    async def test_by_primary_code(self, mock_df):
        data = await call_tool(mcp, "summarize_work_orders", {"equipment_id": "CWC04013"})
        assert "by_primary_code" in data
        assert data["by_primary_code"]["MT010"] == 1
        assert data["by_primary_code"]["MT001"] == 1
        assert data["by_primary_code"]["MT013"] == 1

    @pytest.mark.anyio
    async def test_by_collection(self, mock_df):
        data = await call_tool(mcp, "summarize_work_orders", {"equipment_id": "CWC04013"})
        assert "by_collection" in data
        assert data["by_collection"]["compressor"] == 2
        assert data["by_collection"]["motor"] == 1

    @pytest.mark.anyio
    async def test_date_range_narrows_summary(self, mock_df):
        data = await call_tool(
            mcp,
            "summarize_work_orders",
            {"equipment_id": "CWC04013", "end_date": "2017-09-01"},
        )
        assert data["total_work_orders"] == 2

    @requires_wo_data
    @pytest.mark.anyio
    async def test_integration_summary_cwc04013(self):
        data = await call_tool(
            mcp,
            "summarize_work_orders",
            {"equipment_id": "CWC04013", "start_date": "2017-01-01", "end_date": "2018-01-01"},
        )
        assert "total_work_orders" in data
        assert "by_primary_code" in data
        assert data["total_work_orders"] > 0
