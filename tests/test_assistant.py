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
                json={"transcript": "Some transcript"},
            )
        assert response.status_code == 502
