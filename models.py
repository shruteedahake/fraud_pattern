from typing import Optional
from pydantic import BaseModel


class VendorRedFlagRequest(BaseModel):
    alert_type: str
    severity: str
    title: str
    explanation: str
    triggering_logic: Optional[str] = None
    related_claim_ids: Optional[str] = None


class FraudRiskFlagOutputRequest(BaseModel):
    risk_flag: str
    reason: Optional[str] = None
