#!/bin/bash
# Verify all AssetOpsBench MCP servers can be imported and list their tools.
#
# MCP servers use stdio transport and are spawned on-demand by clients
# (Claude Desktop, plan-execute runner, etc.).  This script confirms each
# server module is correctly installed and reports the tools it exposes.
#
# Usage:
#   ./scripts/start_servers.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0

check_server() {
    local label="$1"
    local module="$2"

    local output
    if output=$(uv run python -c "
from ${module} import mcp
tools = mcp._tool_manager.list_tools()
print(','.join(t.name for t in tools))
" 2>&1); then
        echo "  [OK] $label"
        echo "       tools: $output"
        ((PASS++)) || true
    else
        echo "  [FAIL] $label"
        echo "$output" | sed 's/^/         /'
        ((FAIL++)) || true
    fi
}

echo "================================================"
echo " AssetOpsBench MCP server check"
echo "================================================"
echo ""

check_server "utilities-mcp-server" "servers.utilities.main"
check_server "iot-mcp-server"       "servers.iot.main"
check_server "fmsr-mcp-server"      "servers.fmsr.main"
check_server "tsfm-mcp-server"      "servers.tsfm.main"
check_server "wo-mcp-server"        "servers.wo.main"

echo ""
echo "================================================"
echo " $PASS passed  |  $FAIL failed"
echo "================================================"
echo ""
echo "To run a server (started on-demand by the client):"
echo "  uv run utilities-mcp-server"
echo "  uv run iot-mcp-server"
echo "  uv run fmsr-mcp-server"
echo "  uv run tsfm-mcp-server"
echo "  uv run wo-mcp-server"
echo ""
echo "To run the plan-execute client across all servers:"
echo "  uv run plan-execute \"<your question>\""

[[ $FAIL -eq 0 ]]
