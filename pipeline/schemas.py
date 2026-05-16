from pydantic import BaseModel, Field
from typing import Optional


class SLAClause(BaseModel):
    # --- Performance SLAs ---
    uptime_guarantee: Optional[str] = Field(None, description="Uptime/availability % promised")
    response_time_sla: Optional[str] = Field(None, description="Incident/support response time commitments")
    sla_breach_threshold: Optional[str] = Field(None, description="The point at which a penalty triggers, e.g. 'uptime < 99.5% in any calendar month'")
    sla_measurement_period: Optional[str] = Field(None, description="How/when SLA is measured, e.g. 'calculated monthly', 'rolling 12 months'")

    # --- Penalty clauses (separated by type) ---
    penalty_uptime_breach: Optional[str] = Field(None, description="Service credit or cash penalty for uptime SLA breach")
    penalty_late_delivery: Optional[str] = Field(None, description="Liquidated damages or penalty for late delivery/milestones")
    penalty_termination_fee: Optional[str] = Field(None, description="Fee for early termination or cancellation")
    penalty_late_payment: Optional[str] = Field(None, description="Interest or fee on overdue invoices")
    penalty_data_breach: Optional[str] = Field(None, description="Fine or liability for data breach or security incident")

    # --- Monetary summary fields (for querying) ---
    penalty_has_monetary: Optional[bool] = Field(None, description="True if any cash/monetary penalty exists (not just service credits)")
    penalty_max_amount: Optional[str] = Field(None, description="Largest single penalty amount mentioned, e.g. '$50,000'")
    penalty_currency: Optional[str] = Field(None, description="Currency of penalties, e.g. USD, GBP, EUR")
    service_credit_cap: Optional[str] = Field(None, description="Maximum service credits available, e.g. 'not to exceed 30% of monthly fees'")

    # --- Contract mechanics ---
    renewal_terms: Optional[str] = Field(None, description="Auto-renewal and notice period clauses")
    termination_clause: Optional[str] = Field(None, description="Termination conditions and notice periods")
    liability_cap: Optional[str] = Field(None, description="Limitation of liability cap or maximum damages")
    governing_law: Optional[str] = Field(None, description="Jurisdiction and governing law")
    dispute_resolution: Optional[str] = Field(None, description="Arbitration or mediation clauses")


class ExtractionResult(BaseModel):
    contract_id: str
    file_path: str
    status: str  # "success" | "partial" | "failed"
    sla: SLAClause
    raw_response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
