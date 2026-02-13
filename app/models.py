from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


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
        min_length=10,            # At least 10 chars — a real call is longer
        max_length=50_000,        # 50K chars max — prevents abuse/DOS
        description="The phone call transcript text to analyze",
        json_schema_extra={
            "examples": [
                "Hi, this is Sarah Cohen, born 03/12/1988. I need to book an "
                "appointment because I've had chest pain for two days. Please "
                "call me back at 310-555-2211."
            ]
        },
    )

    @model_validator(mode="after")
    def validate_transcript_content(self) -> "TranscriptRequest":
        """Multi-layer validation to reject non-transcript inputs.

        Each check is a cheap heuristic that runs BEFORE we spend money on
        an LLM call. Defense in depth — multiple weak signals combine for
        strong rejection.
        """
        text = self.transcript

        # 1) Blank check
        if not text.strip():
            raise ValueError("Transcript cannot be blank")

        # 2) Alpha ratio — real transcripts are mostly letters.
        #    Code, binary, or random strings have lots of special chars.
        non_space = [c for c in text if not c.isspace()]
        if non_space:
            alpha_ratio = sum(1 for c in non_space if c.isalpha()) / len(non_space)
            if alpha_ratio < 0.4:
                raise ValueError(
                    "Input doesn't appear to be natural language text "
                    "(too many special characters)."
                )

        # 3) Code pattern detection — reject programming languages.
        #    These patterns almost never appear in phone call transcripts.
        import re
        code_patterns = [
            r"\b(?:def|class|function|import|require|const|let|var)\s+\w+",  # Declarations
            r"[{};]\s*$",                            # Curly braces / semicolons at line ends
            r"\b(?:if|for|while)\s*\(.*\)\s*[{:]",   # Control flow with parens
            r"(?:=>|->|\|\|)\s",                      # Arrow functions, pipe operators
            r"^\s*(?://|#!|/\*)",                     # Comment syntax at line start
            r"</?(?:div|span|html|body|script|head|p|a|img|ul|li|table)\b",  # HTML open tags
            r"</\w+>",                                # HTML close tags (separate signal)
            r"\bself\.\w+\s*=",                       # Python self.x = ...
            r"\breturn\s+\w+",                        # return statements
        ]
        code_matches = sum(
            1 for p in code_patterns if re.search(p, text, re.MULTILINE)
        )
        if code_matches >= 2:
            raise ValueError(
                "Input appears to contain code rather than a phone call transcript."
            )

        # 4) Repetition detection — catch spam/gibberish.
        #    Split into sentences and check if >60% are duplicates.
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]
        if len(sentences) >= 4:
            unique_ratio = len(set(sentences)) / len(sentences)
            if unique_ratio < 0.4:
                raise ValueError(
                    "Input contains excessive repetition and doesn't appear "
                    "to be a genuine transcript."
                )

        # 5) Minimum word count — a real call has at least a few words.
        word_count = len(text.split())
        if word_count < 3:
            raise ValueError(
                "Transcript is too short. Expected at least a few words "
                "of conversation."
            )

        return self


class CallAnalysis(BaseModel):
    # Chain-of-thought: model reasons about the transcript BEFORE assigning
    # confidence. This field is populated by the LLM but excluded from API output.
    reasoning: Optional[str] = Field(
        None,
        description="Internal chain-of-thought reasoning about the transcript. "
        "Explain your classification, note any uncertainties, then justify "
        "the confidence score. This field is excluded from the API response.",
        exclude=True,  # Pydantic will exclude this from .model_dump() by default
    )
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
    speakers: list[str] = Field(
        default_factory=list,
        description="List of identified speakers in the conversation "
        "(e.g., ['Caller', 'Receptionist', 'IVR']). "
        "Empty list for single-speaker transcripts.",
    )
