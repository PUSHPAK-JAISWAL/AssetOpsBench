"""TSFM (Time Series Foundation Model) MCP Server.

Exposes one tool:
  run_tsfm_query  – runs a TSFM agent request and returns the result.

The underlying agent is loaded lazily; if WatsonX credentials are absent
or the tsfmagent package is unavailable the server still starts and returns
an ErrorResult from the tool.
"""

from __future__ import annotations

import logging
import os
from typing import Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")


# ── Result models ─────────────────────────────────────────────────────────────

class ReviewResult(BaseModel):
    status: str
    reasoning: str
    suggestions: str


class TSFMResult(BaseModel):
    answer: str
    reflection: str
    review: ReviewResult
    summary: str


class ErrorResult(BaseModel):
    error: str


# ── Agent init (lazy / graceful) ──────────────────────────────────────────────

def _run_agent(request: str, model_id: int = 0):
    """Import and run the TSFM agent. Raises on failure."""
    from tsfmagent.agents.tsfmagent.tsfm_agent import getTSFMAgent  # type: ignore

    agent = getTSFMAgent(request, llm_model_id=model_id, reflect_step=1, enable_agent_ask=True)
    reaction = agent.run()

    answer = agent.answer.strip() or "Agent failed to generate final answer"
    reflection = (agent.reflections[-1].strip() if agent.reflections else "None")
    review = ReviewResult(
        status=reaction.get("status", "None") if reaction else "None",
        reasoning=reaction.get("reasoning", "None") if reaction else "None",
        suggestions=reaction.get("suggestions", "None") if reaction else "None",
    )
    return TSFMResult(
        answer=answer,
        reflection=reflection,
        review=review,
        summary=(
            "I am TSFM Agent, and I completed my task. The status field denotes the "
            "status of execution. I also received feedback from the review agent, "
            "whose suggestions are included in the review field for further insights."
        ),
    )


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("TSFMAgent")


@mcp.tool()
def run_tsfm_query(
    request: str,
    model_id: int = 0,
) -> Union[TSFMResult, ErrorResult]:
    """Run a Time Series Foundation Model agent query.

    Supports forecasting, anomaly detection, model selection, and regression
    tasks over time-series data. Returns the agent answer, reflection, and
    review from the internal ReAct+Reflexion loop.

    Args:
        request:  Natural-language question or task for the TSFM agent.
        model_id: WatsonX LLM model index (integer, default 0).
    """
    if not request or not request.strip():
        return ErrorResult(error="request is required")

    try:
        return _run_agent(request, model_id=model_id)
    except Exception as exc:
        logger.error("TSFM agent failed: %s", exc)
        return ErrorResult(error=str(exc))


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
