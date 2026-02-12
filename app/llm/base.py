import json
import re
from abc import ABC, abstractmethod

from app.models import CallAnalysis


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to the LLM and return the raw response text."""
        ...

    async def analyze(self, transcript: str) -> CallAnalysis:
        """Analyze a phone call transcript and return structured data."""
        from app.prompt import SYSTEM_PROMPT, build_user_prompt

        raw = await self._call_llm(SYSTEM_PROMPT, build_user_prompt(transcript))
        return self._parse_response(raw)

    @staticmethod
    def _parse_response(raw: str) -> CallAnalysis:
        """Parse the LLM's raw JSON response into a CallAnalysis object."""
        # Strip markdown code fences if present
        cleaned = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

        data = json.loads(cleaned)
        return CallAnalysis.model_validate(data)
