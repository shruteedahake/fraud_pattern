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
        cur.execute("SELECT * FROM fraud_flags WHERE claim_id = %s ORDER BY id DESC", (claim_id,))
        flags = row_to_dict(cur.fetchall())
        return signals, flags
    finally:
        conn.close()


def detect_fraud_patterns(claim_id: str, vendor_id: Optional[str] = None) -> dict:
    claim = _get_claim(claim_id)
    if not claim:
        raise ValueError(f"Claim {claim_id} not found")

    signals, flags = _get_signals_and_flags(claim_id)
    # print(signals)
    # print(flags)

    llm = _get_llm()
    prompt = f"""
You are a fraud-pattern detection assistant for an SIU investigator.
Your task is to analyze the given claim details, AI fraud signals, and fraud flags (if available) below, and identify
1-3 known fraud typologies that may apply. Always give the atleast one pattern in the output. Valid typology keys are:
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

    # print("LLM response:", response)
    # print("LLM patterns:", patterns)

    return {
        "claim_id": claim_id,
        "vendor_id": vendor_id,
        "patterns_identified": patterns,
        "vendor_red_flags_written": written_red_flags,
        "fraud_risk_flags_output_written": written_outputs,
    }
