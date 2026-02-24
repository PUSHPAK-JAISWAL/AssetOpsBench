"""Unit and integration tests for the TSFM MCP server."""

from __future__ import annotations

import pytest

from .conftest import call_tool, requires_watsonx


@pytest.fixture(scope="module")
def mcp():
    from mcp.servers.tsfm.main import mcp as _mcp
    return _mcp


# ── Unit tests (no WatsonX required) ─────────────────────────────────────────

def test_run_tsfm_query_empty_request_returns_error(mcp):
    result = call_tool(mcp, "run_tsfm_query", {"request": ""})
    assert "error" in result
    assert result["error"] == "request is required"


def test_run_tsfm_query_whitespace_request_returns_error(mcp):
    result = call_tool(mcp, "run_tsfm_query", {"request": "   "})
    assert "error" in result


# ── Integration tests (requires WatsonX) ─────────────────────────────────────

@requires_watsonx
def test_run_tsfm_query_integration(mcp):
    result = call_tool(
        mcp,
        "run_tsfm_query",
        {"request": "Forecast the next 5 time steps for a simple sine wave."},
    )
    assert "error" not in result
    assert "answer" in result
    assert "review" in result
