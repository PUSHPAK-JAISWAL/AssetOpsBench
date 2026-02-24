"""Shared fixtures and helpers for TSFM MCP server tests."""

from __future__ import annotations

import json
import os

import pytest

requires_watsonx = pytest.mark.skipif(
    not os.environ.get("WATSONX_APIKEY"),
    reason="WATSONX_APIKEY not set",
)


def call_tool(mcp, tool_name: str, args: dict) -> dict:
    """Call an MCP tool and return the parsed JSON result."""
    import asyncio
    from mcp.server.fastmcp import FastMCP

    async def _call():
        result = await mcp._mcp_call_tool(tool_name, args)
        return json.loads(result.content[0].text)

    return asyncio.get_event_loop().run_until_complete(_call())
