"""Tests for the Clinic Phone Assistant.

These tests validate:
- Pydantic model validation
- JSON response parsing
- FastAPI endpoint contracts
- Urgency detection logic via the prompt
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.llm.base import BaseLLMProvider
from app.main import app
from app.models import CallAnalysis, CallIntent, TranscriptRequest, UrgencyLevel


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------

class TestModels:
    def test_valid_call_analysis(self):
        data = {
            "intent": "appointment_booking",
            "name": "John Doe",
            "dob": "1990-05-15",
            "phone": "555-1234",
            "summary": "Wants to book a checkup",
            "urgency": "low",
            "confidence": 0.95,
        }
        result = CallAnalysis.model_validate(data)
        assert result.intent == CallIntent.APPOINTMENT_BOOKING
        assert result.name == "John Doe"
        assert result.urgency == UrgencyLevel.LOW

    def test_optional_fields_null(self):
        data = {
            "intent": "general_inquiry",
            "name": None,
            "dob": None,
            "phone": None,
            "summary": "Asking about clinic hours",
            "urgency": "low",
            "confidence": 0.8,
        }
        result = CallAnalysis.model_validate(data)
        assert result.name is None
        assert result.dob is None
        assert result.phone is None

    def test_invalid_intent_rejected(self):
        data = {
            "intent": "pizza_order",
            "summary": "Invalid",
            "urgency": "low",
            "confidence": 0.5,
        }
        with pytest.raises(Exception):
            CallAnalysis.model_validate(data)

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            CallAnalysis(
                intent=CallIntent.GENERAL_INQUIRY,
                summary="Test",
                urgency=UrgencyLevel.LOW,
                confidence=1.5,
            )

    def test_transcript_request_non_empty(self):
        with pytest.raises(Exception):
            TranscriptRequest(transcript="")


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Test the multi-layer input validation on TranscriptRequest."""

    def test_valid_transcript_accepted(self):
        req = TranscriptRequest(
            transcript="Hi, this is John. I need to schedule an appointment."
        )
        assert req.transcript.startswith("Hi")

    def test_too_short_rejected(self):
        with pytest.raises(Exception, match="least"):
            TranscriptRequest(transcript="Hi there")  # < 10 chars after min_length

    def test_code_input_rejected(self):
        code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total

class ShoppingCart:
    def __init__(self):
        self.items = []
"""
        with pytest.raises(Exception, match="code"):
            TranscriptRequest(transcript=code)

    def test_html_input_rejected(self):
        html = """
<div class="container">
    <span>Hello</span>
    <script>alert('xss')</script>
</div>
"""
        with pytest.raises(Exception, match="code"):
            TranscriptRequest(transcript=html)

    def test_special_chars_rejected(self):
        garbage = "!@#$%^&*(){}[]|\\/<>~`+=:;!@#$%^&*()"
        with pytest.raises(Exception, match="natural language"):
            TranscriptRequest(transcript=garbage)

    def test_repetitive_spam_rejected(self):
        spam = "Buy now. " * 20  # Same sentence repeated 20 times
        with pytest.raises(Exception, match="repetition"):
            TranscriptRequest(transcript=spam)

    def test_few_words_rejected(self):
        with pytest.raises(Exception):
            TranscriptRequest(transcript="hello there")  # Only 2 words

    def test_valid_multi_speaker_accepted(self):
        transcript = (
            "Receptionist: Good morning, City Health Clinic.\n"
            "Caller: Hi, I need to reschedule my appointment.\n"
            "Receptionist: Sure, what's your name?\n"
            "Caller: It's James Park."
        )
        req = TranscriptRequest(transcript=transcript)
        assert "James Park" in req.transcript

    def test_transcript_with_numbers_accepted(self):
        """Transcripts naturally contain phone numbers, dates, etc."""
        transcript = (
            "Hi, my name is Lisa Chen, born 05/15/1992. "
            "My phone is 555-123-4567. I need to refill prescription #RX12345."
        )
        req = TranscriptRequest(transcript=transcript)
        assert req.transcript == transcript


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

class TestResponseParsing:
    def test_parse_clean_json(self):
        raw = json.dumps({
            "intent": "prescription_refill",
            "name": "David Levi",
            "dob": "1975-01-05",
            "phone": "555-0199",
            "summary": "Refill blood pressure medication",
            "urgency": "low",
            "confidence": 0.97,
        })
        result = BaseLLMProvider._parse_response(raw)
        assert result.intent == CallIntent.PRESCRIPTION_REFILL
        assert result.name == "David Levi"

    def test_parse_json_with_code_fences(self):
        raw = '```json\n{"intent": "billing_question", "name": null, "dob": null, "phone": "555-9999", "summary": "Question about a charge", "urgency": "low", "confidence": 0.9}\n```'
        result = BaseLLMProvider._parse_response(raw)
        assert result.intent == CallIntent.BILLING_QUESTION

    def test_parse_json_with_bare_fences(self):
        raw = '```\n{"intent": "general_inquiry", "summary": "Clinic hours", "urgency": "low", "confidence": 0.85}\n```'
        result = BaseLLMProvider._parse_response(raw)
        assert result.intent == CallIntent.GENERAL_INQUIRY


# ---------------------------------------------------------------------------
# API endpoint tests (mocked LLM)
# ---------------------------------------------------------------------------

MOCK_ANALYSIS = CallAnalysis(
    intent=CallIntent.URGENT_MEDICAL_ISSUE,
    name="Sarah Cohen",
    dob="1988-03-12",
    phone="310-555-2211",
    summary="Chest pain for two days",
    urgency=UrgencyLevel.HIGH,
    confidence=0.95,
)


@pytest.mark.asyncio
class TestAPI:
    async def test_health_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @patch("app.main.get_provider")
    async def test_analyze_endpoint(self, mock_get_provider):
        mock_provider = AsyncMock()
        mock_provider.analyze.return_value = MOCK_ANALYSIS
        mock_get_provider.return_value = mock_provider

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/analyze",
                json={"transcript": "Hi, this is Sarah Cohen..."},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "urgent_medical_issue"
        assert data["name"] == "Sarah Cohen"
        assert data["urgency"] == "high"

    @patch("app.main.get_provider")
    async def test_analyze_empty_transcript_rejected(self, mock_get_provider):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/analyze", json={"transcript": ""})
        assert response.status_code == 422

    @patch("app.main.get_provider")
    async def test_analyze_llm_failure_returns_502(self, mock_get_provider):
        mock_provider = AsyncMock()
        mock_provider.analyze.side_effect = Exception("API timeout")
        mock_get_provider.return_value = mock_provider

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/analyze",
                json={"transcript": "Hi, I need to schedule an appointment for next week please."},
            )
        assert response.status_code == 502
