"""FMSR (Failure Mode and Sensor Reasoning) MCP Server.

Exposes three tools:
  get_failure_modes               – lists failure modes for an asset
  get_sensors                     – lists available sensors for an asset
  get_failure_mode_sensor_mapping – returns bidirectional FM↔sensor relevancy mapping

For chillers and AHUs the first two tools return curated hardcoded lists.
For any other asset type the LLM is queried as a fallback.
The mapping tool always calls the LLM to determine per-pair relevancy.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser, NumberedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from mcp.server.fastmcp import FastMCP

load_dotenv()

_log_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("fmsr-mcp-server")


# ── Hardcoded asset data ──────────────────────────────────────────────────────

_CHILLER_SENSORS = [
    "Chiller % Loaded",
    "Chiller Efficiency",
    "Condenser Water Flow",
    "Condenser Water Supply To Chiller Temperature",
    "Liquid Refrigerant Evaporator Temperature",
    "Power Input",
    "Return Temperature",
    "Schedule",
    "Supply Temperature",
    "Tonnage",
]

_ASSET_FAILURE_MODES: dict[str, list[str]] = {
    "chiller": [
        "Compressor Overheating: Failed due to Normal wear, overheating",
        "Heat Exchangers: Fans: Degraded motor or worn bearing due to Normal use",
        "Evaporator Water side fouling",
        "Condenser Water side fouling",
        "Condenser Improper water side flow rate",
        "Purge Unit Excessive purge",
        "Refrigerant Operated Control Valve Failed spring",
    ],
    "ahu": [
        "Pressure Regulators Diaphragm failure",
        "Steam Heating Coils Air side fouling",
        "Belts or sheaves Wear",
        "Improper switch position",
        "Solenoid Valves Bound due to hardened grease",
    ],
}


# ── Output parsers ────────────────────────────────────────────────────────────

class _SensorParser(NumberedListOutputParser):
    """Strips trailing 'Sensor(s)' suffix from numbered-list LLM output."""

    def parse(self, text: str) -> List[str]:
        sensors = re.findall(self.pattern, text)
        return [
            s.replace(" Sensors", "").replace(" sensors", "")
             .replace(" Sensor", "").replace(" sensor", "")
            for s in sensors
        ]


class _RelevancyParser(JsonOutputParser):
    """Parses a 3-line LLM response into {answer, reason, temporal_behavior}."""

    def parse_result(self, result, *, partial: bool = False):
        text = result[0].text.strip()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines and lines[0].lower().startswith("yes"):
            answer = "Yes"
        elif lines and lines[0].lower().startswith("no"):
            answer = "No"
        else:
            answer = "Unknown"
        reason = lines[1] if len(lines) >= 2 else "Unknown"
        temporal = lines[2] if (answer == "Yes" and len(lines) >= 3) else "Unknown"
        return {"answer": answer, "reason": reason, "temporal_behavior": temporal}


# ── Prompts ───────────────────────────────────────────────────────────────────

_asset2fm_prompt = ChatPromptTemplate.from_template(
    "What are different failure modes for asset {asset_name}?\n"
    "Your response should be a numbered list with each failure mode on a new line. "
    "Please only list the failure mode name.\n"
    "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
)

_asset2sensor_prompt = ChatPromptTemplate.from_template(
    "What are the sensors that can be installed in the asset {asset_name} "
    "for monitoring the performance?\n"
    "Your response should be a numbered list with each sensor name on a new line. "
    "Please only list the sensor name.\n"
    "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
)

_relevancy_prompt = ChatPromptTemplate.from_template(
    "For the asset {asset_name}, if the failure {failure_mode} occurs, "
    "can sensor {sensor} help monitor or detect the failure for {asset_name}?\n"
    "Provide the answer in the first line and reason in the second line. "
    "If the answer is Yes, provide the temporal behaviour of the sensor "
    "when the failure occurs in the third line."
)


# ── LLM + chains (lazy init; graceful degradation if creds are absent) ────────

def _build_chains():
    from langchain_ibm import ChatWatsonx

    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url=os.environ["WATSONX_URL"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        api_key=os.environ["WATSONX_APIKEY"],
        params={"max_tokens": 10000, "temperature": 0.0},
    )
    llm_with_retry = llm.with_retry(stop_after_attempt=3)
    return (
        _asset2fm_prompt | llm_with_retry | NumberedListOutputParser(),
        _asset2sensor_prompt | llm_with_retry | _SensorParser(),
        _relevancy_prompt | llm | _RelevancyParser(),
    )


try:
    _asset2fm_chain, _asset2sensor_chain, _relevancy_chain = _build_chains()
    _llm_available = True
except Exception as _e:
    logger.error("WatsonX LLM unavailable: %s", _e)
    _asset2fm_chain = _asset2sensor_chain = _relevancy_chain = None
    _llm_available = False


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("FMSRAgent")


@mcp.tool()
def get_failure_modes(asset_name: str) -> str:
    """Returns a list of known failure modes for the given asset.
    For chillers and AHUs returns a curated list. For other assets queries the LLM."""
    asset_key = re.sub(r"\d+", "", asset_name).strip().lower()
    if not asset_key or asset_key == "none":
        return json.dumps({"error": "asset_name is required"})

    if asset_key in _ASSET_FAILURE_MODES:
        return json.dumps({
            "asset_name": asset_name,
            "failure_modes": _ASSET_FAILURE_MODES[asset_key],
        })

    if not _llm_available:
        return json.dumps({"error": "LLM unavailable and asset not in local database"})

    try:
        result = _asset2fm_chain.invoke({"asset_name": asset_name})
        return json.dumps({"asset_name": asset_name, "failure_modes": result})
    except Exception as exc:
        logger.error("asset2fm_chain failed: %s", exc)
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_sensors(asset_name: str) -> str:
    """Returns a list of available sensors for the given asset.
    For chillers returns a curated list. For other assets queries the LLM."""
    asset_name = re.split(r"\n", asset_name)[0].strip()
    if not asset_name:
        return json.dumps({"error": "asset_name is required"})

    if asset_name.lower().startswith("chiller"):
        return json.dumps({
            "asset_name": asset_name,
            "sensors": [f"{asset_name} {s}" for s in _CHILLER_SENSORS],
        })

    if not _llm_available:
        return json.dumps({"error": "LLM unavailable and asset not in local database"})

    try:
        result = _asset2sensor_chain.invoke({"asset_name": asset_name})
        return json.dumps({"asset_name": asset_name, "sensors": result})
    except Exception as exc:
        logger.error("asset2sensor_chain failed: %s", exc)
        return json.dumps({"error": str(exc)})


@mcp.tool()
def get_failure_mode_sensor_mapping(
    asset_name: str,
    failure_modes: List[str],
    sensors: List[str],
) -> str:
    """For each (failure_mode, sensor) pair determines whether the sensor can detect
    the failure. Returns a bidirectional mapping (fm→sensors, sensor→fms) plus
    the full per-pair relevancy details."""
    if not asset_name:
        return json.dumps({"error": "asset_name is required"})
    if not failure_modes:
        return json.dumps({"error": "failure_modes list is required"})
    if not sensors:
        return json.dumps({"error": "sensors list is required"})
    if not _llm_available:
        return json.dumps({"error": "LLM unavailable"})

    batches = [
        {"asset_name": asset_name, "failure_mode": fm, "sensor": s}
        for s in sensors
        for fm in failure_modes
    ]

    try:
        generations = _relevancy_chain.batch(batches, config={"max_concurrency": 1})
    except Exception as exc:
        logger.error("relevancy_chain.batch failed: %s", exc)
        return json.dumps({"error": str(exc)})

    full_relevancy = []
    fm2sensor: dict = {}
    sensor2fm: dict = {}
    for batch, gen in zip(batches, generations):
        entry = {
            **batch,
            "relevancy_answer": gen["answer"],
            "relevancy_reason": gen["reason"],
            "temporal_behavior": gen["temporal_behavior"],
        }
        full_relevancy.append(entry)
        if "yes" in gen["answer"].lower():
            fm2sensor.setdefault(batch["failure_mode"], []).append(batch["sensor"])
            sensor2fm.setdefault(batch["sensor"], []).append(batch["failure_mode"])

    return json.dumps({
        "metadata": {
            "asset_name": asset_name,
            "failure_modes": failure_modes,
            "sensors": sensors,
        },
        "fm2sensor": fm2sensor,
        "sensor2fm": sensor2fm,
        "full_relevancy": full_relevancy,
    })


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
