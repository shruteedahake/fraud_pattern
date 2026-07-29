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
            vendor_id, body.alert_type, body.severity, body.title, body.explanation,
            body.triggering_logic, body.related_claim_ids,
        )
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
        return handler.write_fraud_risk_flag_output(vendor_id, body.risk_flag, body.reason)
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
