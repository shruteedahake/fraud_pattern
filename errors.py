/home/JarvisClaims/Shrutee/Jarvis_claims_agents_v3_siu/SIUAgents/MCP/main.py:281: DeprecationWarning: 
        on_event is deprecated, use lifespan event handlers instead.

        Read more about it in the
        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
        
  @app.on_event("startup")
INFO:     Started server process [26504]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9998 (Press CTRL+C to quit)
INFO:     127.0.0.1:57892 - "POST /fraud-pattern/detect-fraud-patterns HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:39264 - "POST /fraud-pattern/detect-fraud-patterns HTTP/1.1" 404 Not Found
INFO:     127.0.0.1:56632 - "POST /api/fraud_pattern/detect/CLM-2026-5789 HTTP/1.1" 404 Not Found


server.py

"""
server.py — Fraud Pattern Agent
──────────────────────────────────────
LangGraph agent that recomputes and explains the aggregate fraud risk score
for a claim, for SIU investigators.

Port: 9005
MCP : http://localhost:9000/api/v1/fraud_pattern/mcp

Run:
    py -3 server.py
"""

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Annotated, TypedDict

import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai.chat_models import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("fraud_pattern_agent")

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY", "")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "")
MCP_URL = os.getenv("MCP_URL", "http://localhost:9998/api/v1/fraud_pattern/mcp")
AGENT_PORT = int(os.getenv("AGENT_PORT", "9005"))

config_mcp_server = {
    "fraud_pattern_mcp": {
        "url": MCP_URL,
        "transport": "streamable_http",
        "timeout": timedelta(seconds=120),
        "sse_read_timeout": timedelta(seconds=600),
    }
}

app = FastAPI(title="Fraud Pattern Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def router(state: State):
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    if isinstance(last, AIMessage) and last.content:
        if "Continue" in last.content:
            return "tools"
        if "End" in last.content:
            return "End"
    return "End"


_FALLBACK_PROMPT = """
You are the Fraud Pattern Agent for an insurance claims platform, assisting
an SIU (Special Investigation Unit) investigator.

Given a claim_id (and optionally a vendor_id), your workflow is:

1. Call detect_fraud_patterns with the claim_id (and vendor_id if provided).
   This will:
   - Read the claim, AI fraud signals, and fraud flags.
   - Use AI analysis to identify 0-3 known fraud typologies (staged_loss,
     inflated_estimate, rapid_repeat_claim, vendor_collusion) with severity.
   - Write vendor_red_flags rows for Medium/High/Critical patterns (if a
     vendor_id was given) and fraud_risk_flags_output rows for all
     identified patterns.
2. Summarize for the investigator the identified typologies, their
   severities, and any flags that were written.

You may also use get_vendor_red_flags and get_fraud_risk_flags_output to
inspect existing data on request.

When you have completed the task, end your response with "End".
"""


def load_prompt() -> str:
    if not PHOENIX_ENDPOINT:
        raise RuntimeError("Phoenix not configured")
    from phoenix.client import Client
    client = Client(base_url=PHOENIX_ENDPOINT, api_key=PHOENIX_API_KEY)
    prompt = client.prompts.get(name="fraud_pattern_agent", label="production")
    prompt_set = prompt._template["messages"]
    system_msg = next(
        (item["content"][0]["text"] for item in prompt_set if item.get("role") == "system"),
        None,
    )
    if not system_msg:
        raise ValueError("System prompt is empty or missing in Phoenix")
    return system_msg


def create_graph(model, tools, prompt):
    graph_builder = StateGraph(State)
    llm_with_tools = model.bind_tools(tools)

    async def agent_node(state: State):
        messages = state["messages"]
        all_messages = [SystemMessage(content=prompt)] + messages
        message = await llm_with_tools.ainvoke(all_messages)
        return {"messages": [message]}

    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", router, {"tools": "tools", "End": END})
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile()


async def get_tools():
    client = MultiServerMCPClient(config_mcp_server)
    tools = await client.get_tools()
    logger.info("Tools loaded from MCP: %s", [t.name for t in tools])
    return tools


async def stream_graph(graph, initial_state, config):
    async for event in graph.astream_events(initial_state, config=config, version="v2"):
        kind = event.get("event", "")

        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield f"data: {chunk.content}\n\n"

        elif kind == "on_tool_start":
            tool_name = event.get("name", "unknown_tool")
            yield f"data: [Tool: {tool_name}] Starting...\n\n"

        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown_tool")
            yield f"data: [Tool: {tool_name}] Done\n\n"


@app.post("/chat")
async def chat_stream(request: Request):
    load_dotenv(find_dotenv())

    tools = await get_tools()

    model = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    try:
        system_prompt = load_prompt()
    except Exception as e:
        logger.warning("Phoenix prompt load failed (%s) — using fallback prompt", e)
        system_prompt = _FALLBACK_PROMPT

    body = await request.json()
    user_message = body.get("message", "Check for known fraud patterns on this claim")

    graph = create_graph(model=model, tools=tools, prompt=system_prompt)

    async def generate():
        start = time.time()
        last_event_at = start
        last_tool = None
        try:
            async for event in stream_graph(
                graph=graph,
                initial_state={"messages": [user_message]},
                config={"recursion_limit": 250},
            ):
                last_event_at = time.time()
                if isinstance(event, str) and event.startswith("data: [Tool:"):
                    try:
                        last_tool = event.split("[Tool:", 1)[1].split("]", 1)[0]
                    except Exception:
                        pass
                yield event
        except BaseException as e:
            elapsed = time.time() - start
            since_last = time.time() - last_event_at
            err = {
                "exception_class": type(e).__name__,
                "message": str(e),
                "elapsed_total_seconds": round(elapsed, 2),
                "seconds_since_last_event": round(since_last, 2),
                "last_tool_invoked": last_tool,
                "traceback": traceback.format_exc(),
                "timestamp_utc": datetime.utcnow().isoformat(),
            }
            logger.error("AGENT_ERROR %s", json.dumps(err, default=str))
            try:
                yield f"data: [AGENT_ERROR] {json.dumps(err, default=str)}\n\n"
            except Exception:
                pass
            import asyncio
            if isinstance(e, asyncio.CancelledError):
                raise

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "fraud_pattern_agent"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)


fraud_pattern_router.py

"""
fraud_pattern_router.py
─────────────────────────
FastAPI routes for the Fraud Pattern MCP.

Tool / Endpoint map:
  get_vendor_red_flags          GET  /api/fraud_pattern/red_flags/{vendor_id}
  write_vendor_red_flag         POST /api/fraud_pattern/red_flags/{vendor_id}
  get_fraud_risk_flags_output   GET  /api/fraud_pattern/risk_flags/{vendor_id}
  write_fraud_risk_flag_output  POST /api/fraud_pattern/risk_flags/{vendor_id}
  detect_fraud_patterns         POST /api/fraud_pattern/detect/{claim_id}
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException

from fraud_pattern_mcp import handler
from fraud_pattern_mcp.models import FraudRiskFlagOutputRequest, VendorRedFlagRequest

log = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/api/fraud_pattern/red_flags/{vendor_id}",
    operation_id="get_vendor_red_flags",
    summary="Get vendor red flags",
    tags=["FraudPattern"],
)
def get_vendor_red_flags(vendor_id: str):
    """Returns all vendor_red_flags rows for the given vendor_id."""
    return handler.get_vendor_red_flags(vendor_id)


@router.post(
    "/api/fraud_pattern/red_flags/{vendor_id}",
    operation_id="write_vendor_red_flag",
    summary="Write a vendor red flag",
    tags=["FraudPattern"],
)
def write_vendor_red_flag(vendor_id: str, body: VendorRedFlagRequest):
    """Inserts a new vendor_red_flags row for the given vendor_id."""
    try:
        return handler.write_vendor_red_flag(
            vendor_id, body.claim_id, body.alert_type, body.severity, body.title, body.explanation,
            body.triggering_logic, body.related_claim_ids,
        )
        # return handler.write_vendor_red_flag(
        #     vendor_id, body.alert_type, body.severity, body.title, body.explanation,
        #     body.triggering_logic, body.related_claim_ids,
        # )
    except Exception as e:
        log.exception("write_vendor_red_flag error")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/fraud_pattern/risk_flags/{vendor_id}",
    operation_id="get_fraud_risk_flags_output",
    summary="Get fraud risk flags output for a vendor",
    tags=["FraudPattern"],
)
def get_fraud_risk_flags_output(vendor_id: str):
    """Returns all fraud_risk_flags_output rows for the given vendor_id."""
    return handler.get_fraud_risk_flags_output(vendor_id)


@router.post(
    "/api/fraud_pattern/risk_flags/{vendor_id}",
    operation_id="write_fraud_risk_flag_output",
    summary="Write a fraud risk flag output for a vendor",
    tags=["FraudPattern"],
)
def write_fraud_risk_flag_output(vendor_id: str, body: FraudRiskFlagOutputRequest):
    """Inserts a new fraud_risk_flags_output row for the given vendor_id."""
    try:
        return handler.write_fraud_risk_flag_output(vendor_id, body.claim_id, body.risk_flag, body.reason)
        # return handler.write_fraud_risk_flag_output(vendor_id, body.risk_flag, body.reason)
    except Exception as e:
        log.exception("write_fraud_risk_flag_output error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/fraud_pattern/detect/{claim_id}",
    operation_id="detect_fraud_patterns",
    summary="Detect known fraud typologies for a claim",
    tags=["FraudPattern"],
)
def detect_fraud_patterns(claim_id: str, vendor_id: Optional[str] = None):
    """
    Reads the claim, ai_fraud_signals, and fraud_flags for claim_id; uses an
    LLM to identify 0-3 known fraud typologies. For Medium/High/Critical
    patterns with a vendor_id, writes a vendor_red_flags row. Always writes a
    fraud_risk_flags_output row per identified pattern.
    """
    try:
        return handler.detect_fraud_patterns(claim_id, vendor_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("detect_fraud_patterns error")
        raise HTTPException(status_code=500, detail=str(e))

handler.py

"""
handler.py — Fraud Pattern
────────────────────────────
AI-assisted known fraud-typology detection for a claim (and optionally a
vendor), writing vendor_red_flags and fraud_risk_flags_output rows.
"""

import json
import logging
import os
import sys
from typing import Optional 

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

from db import get_db_connection, row_to_dict  # noqa: E402
from langchain_openai.chat_models import AzureChatOpenAI  # noqa: E402

log = logging.getLogger(__name__)

KNOWN_PATTERNS = ["staged_loss", "inflated_estimate", "rapid_repeat_claim", "vendor_collusion"]


def _get_llm():
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def get_vendor_red_flags(vendor_id: str) -> list:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM vendor_red_flags WHERE vendor_id = %s ORDER BY id DESC", (vendor_id,))
        return row_to_dict(cur.fetchall())
    finally:
        conn.close()


# def write_vendor_red_flag(vendor_id: str, alert_type: str, severity: str, title: str,
#                            explanation: str, triggering_logic: Optional[str] = None,
#                            related_claim_ids: Optional[str] = None) -> dict:
def write_vendor_red_flag(vendor_id: str, claim_id:str, alert_type: str, severity: str, title: str,
                           explanation: str, triggering_logic: Optional[str] = None,
                           related_claim_ids: Optional[str] = None) -> dict:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            # """
            # INSERT INTO vendor_red_flags (vendor_id, alert_type, severity, title, explanation,
            #                                triggering_logic, related_claim_ids)
            # VALUES (%s,%s,%s,%s,%s,%s,%s)
            # """,
            """
            INSERT INTO vendor_red_flags (vendor_id, claim_id, alert_type, severity, title, explanation,
                                            triggering_logic, related_claim_ids)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (vendor_id, claim_id, alert_type, severity, title, explanation, triggering_logic, related_claim_ids),
        )
        conn.commit()
        return {
            "id": cur.lastrowid, "vendor_id": vendor_id, "claim_id": claim_id, "alert_type": alert_type,
            "severity": severity, "title": title, "explanation": explanation,
            "triggering_logic": triggering_logic, "related_claim_ids": related_claim_ids,
        }
    finally:
        conn.close()


def get_fraud_risk_flags_output(vendor_id: str) -> list:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM fraud_risk_flags_output WHERE vendor_id = %s ORDER BY id DESC", (vendor_id,))
        return row_to_dict(cur.fetchall())
    finally:
        conn.close()


# def write_fraud_risk_flag_output(vendor_id: str, risk_flag: str, reason: Optional[str] = None) -> dict:
def write_fraud_risk_flag_output(vendor_id: str, claim_id:str, risk_flag: str, reason: Optional[str] = None) -> dict:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        # cur.execute(
        #     "INSERT INTO fraud_risk_flags_output (vendor_id, risk_flag, reason) VALUES (%s,%s,%s)",
        #     (vendor_id, risk_flag, reason),
        # )
        cur.execute(
            "INSERT INTO fraud_risk_flags_output (vendor_id, claim_id, risk_flag, reason) VALUES (%s,%s,%s,%s)",
            (vendor_id, claim_id, risk_flag, reason),
        )
        conn.commit()
        return {"id": cur.lastrowid, "vendor_id": vendor_id, "claim_id": claim_id, "risk_flag": risk_flag, "reason": reason}
    finally:
        conn.close()


def _get_claim(claim_id: str) -> Optional[dict]:
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM claims WHERE claim_number = %s", (claim_id,))
        row = cur.fetchone()
        if row:
            return row_to_dict(row)
        if claim_id.isdigit():
            cur.execute("SELECT * FROM claims WHERE id = %s", (int(claim_id),))
            return row_to_dict(cur.fetchone())
        return None
    finally:
        conn.close()


def _get_signals_and_flags(claim_id: str):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM ai_fraud_signals WHERE claim_id = %s ORDER BY id DESC", (claim_id,))
        signals = row_to_dict(cur.fetchall())
        cur.execute("SELECT * cROM fraud_flags WHERE claim_id = %s ORDER BY id DESC", (claim_id,))
        flags = row_to_dict(cur.fetchall())
        return signals, flags
    finally:
        conn.close()


def detect_fraud_patterns(claim_id: str, vendor_id: Optional[str] = None) -> dict:
    claim = _get_claim(claim_id)
    if not claim:
        raise ValueError(f"Claim {claim_id} not found")

    signals, flags = _get_signals_and_flags(claim_id)
    print(signals)
    print(flags)

    llm = _get_llm()
    prompt = f"""
You are a fraud-pattern detection assistant for an SIU investigator.
Your task is to analyze the given claim details, AI fraud signals, and fraud flags (if available) below, and identify
0-3 known fraud typologies that may apply. Valid typology keys are:
{KNOWN_PATTERNS}

Important Rule- 
1. AI fraud signals may contain MULTIPLE records for the same claim.
2. Fraud flags may be EMPTY.
    - IF fraud flags are missing, rely ONLY on the AI fraud signals + claim data.
    - Do not assume fraud just because flags are missing.

For each identified pattern, provide: "pattern" (one of the keys above),
"severity" (one of "Low", "Medium", "High", "Critical"), "title" (short
human-readable title), and "explanation" (1-2 sentences).

Claim details:
  loss_type: {claim.get('loss_type')}
  short_description: {claim.get('short_description')}
  severity: {claim.get('severity')}
  estimated_cost: {claim.get('estimated_cost')}
  date_of_loss: {claim.get('date_of_loss')}

AI fraud signals (may contain multiple entries): {json.dumps(signals, default=str)}
Fraud flags (may be empty): {json.dumps(flags, default=str)}

Respond with ONLY a JSON object: {{"patterns": [{{"pattern": "...", "severity": "...", "title": "...", "explanation": "..."}}]}}
If nothing notable, return an empty list.
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    try:
        parsed = json.loads(content)
        patterns = parsed.get("patterns", [])
    except Exception:
        log.warning("Could not parse LLM JSON, defaulting to empty: %s", content)
        patterns = []

    written_red_flags = []
    written_outputs = []

    for p in patterns[:3]:
        pattern = p.get("pattern", "unknown_pattern")
        severity = p.get("severity", "Low")
        title = p.get("title", pattern)
        explanation = p.get("explanation", "")

        if severity in ("Medium", "High", "Critical") and vendor_id:
            written_red_flags.append(write_vendor_red_flag(
                vendor_id=vendor_id,
                claim_id=claim_id,
                alert_type=pattern,
                severity=severity,
                title=title,
                explanation=explanation,
                triggering_logic=f"AI-detected pattern '{pattern}' for claim {claim_id}",
                related_claim_ids=json.dumps([claim_id]),
            ))

        written_outputs.append(write_fraud_risk_flag_output(
            vendor_id=vendor_id or "N/A",
            claim_id=claim_id,
            risk_flag=pattern,
            reason=f"{title}: {explanation} (severity={severity})",
        ))

    return {
        "claim_id": claim_id,
        "vendor_id": vendor_id,
        "patterns_identified": patterns,
        "vendor_red_flags_written": written_red_flags,
        "fraud_risk_flags_output_written": written_outputs,
    }








-------------------------------------------

LLM response: content='{"patterns": []}' additional_kwargs={'parsed': None, 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 6, 'prompt_tokens': 435, 'total_tokens': 441, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': None, 'cached_tokens': 0}, 'latency_checkpoint': {'engine_tbt_ms': 0, 'engine_ttft_ms': 318, 'engine_ttlt_ms': 319, 'pre_inference_ms': 269, 'service_tbt_ms': 2, 'service_ttft_ms': 1156, 'service_ttlt_ms': 1166, 'total_duration_ms': 901, 'user_visible_ttft_ms': 887}}, 'model_provider': 'openai', 'model_name': 'gpt-4.1-2025-04-14', 'system_fingerprint': 'fp_3cba29e44e', 'id': 'chatcmpl-E7GD7xRemPOkJCdxVoyWCHxPabxRo', 'service_tier': 'default', 'prompt_filter_results': [{'prompt_index': 0, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'jailbreak': {'detected': False, 'filtered': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}}], 'finish_reason': 'stop', 'logprobs': None, 'content_filter_results': {'hate': {'filtered': False, 'severity': 'safe'}, 'protected_material_code': {'detected': False, 'filtered': False}, 'protected_material_text': {'detected': False, 'filtered': False}, 'self_harm': {'filtered': False, 'severity': 'safe'}, 'sexual': {'filtered': False, 'severity': 'safe'}, 'violence': {'filtered': False, 'severity': 'safe'}}} id='lc_run--019fb217-93f1-7b33-8844-2b4422ab2471-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 435, 'output_tokens': 6, 'total_tokens': 441, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
LLM patterns: []
