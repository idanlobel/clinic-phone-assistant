from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CallIntent(str, Enum):
    APPOINTMENT_BOOKING = "appointment_booking"
    PRESCRIPTION_REFILL = "prescription_refill"
    BILLING_QUESTION = "billing_question"
    URGENT_MEDICAL_ISSUE = "urgent_medical_issue"
    GENERAL_INQUIRY = "general_inquiry"
    INSURANCE_QUESTION = "insurance_question"
    LAB_RESULTS = "lab_results"
    REFERRAL_REQUEST = "referral_request"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TranscriptRequest(BaseModel):
    transcript: str = Field(
        ...,
        min_length=1,
        description="The phone call transcript text to analyze",
        json_schema_extra={
            "examples": [
                "Hi, this is Sarah Cohen, born 03/12/1988. I need to book an "
                "appointment because I've had chest pain for two days. Please "
                "call me back at 310-555-2211."
            ]
        },
    )


class CallAnalysis(BaseModel):
    intent: CallIntent = Field(description="The primary intent of the caller")
    name: Optional[str] = Field(None, description="Caller's full name")
    dob: Optional[str] = Field(
        None, description="Date of birth in ISO format (YYYY-MM-DD)"
    )
    phone: Optional[str] = Field(None, description="Callback phone number")
    summary: str = Field(description="Brief summary of the reason for calling")
    urgency: UrgencyLevel = Field(description="Urgency level of the call")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence in the analysis (0-1)"
    )
