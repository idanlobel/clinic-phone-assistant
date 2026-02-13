import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.models import CallAnalysis

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Send a prompt to the LLM and return the raw response text."""
        ...

    @abstractmethod
    async def _stream_llm(
        self, system_prompt: str, user_message: str
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM one chunk at a time."""
        ...
        yield ""  # pragma: no cover

    async def analyze(self, transcript: str) -> CallAnalysis:
        """Analyze a phone call transcript and return structured data."""
        from app.prompt import SYSTEM_PROMPT, build_user_prompt

        start = time.perf_counter()
        raw = await self._call_llm(SYSTEM_PROMPT, build_user_prompt(transcript))
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "LLM call completed",
            extra={"extra_data": {
                "provider": self.__class__.__name__,
                "duration_ms": round(duration_ms, 2),
                "response_length": len(raw),
            }},
        )

        return self._parse_response(raw)

    async def stream_analyze(self, transcript: str) -> AsyncIterator[str]:
        """Stream the raw LLM response token-by-token."""
        from app.prompt import SYSTEM_PROMPT, build_user_prompt

        start = time.perf_counter()
        chunk_count = 0
        async for chunk in self._stream_llm(
            SYSTEM_PROMPT, build_user_prompt(transcript)
        ):
            chunk_count += 1
            yield chunk

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "LLM stream completed",
            extra={"extra_data": {
                "provider": self.__class__.__name__,
                "duration_ms": round(duration_ms, 2),
                "chunks": chunk_count,
            }},
        )

    @staticmethod
    def _parse_response(raw: str) -> CallAnalysis:
        """Parse the LLM's raw JSON response into a CallAnalysis object."""
        cleaned = raw.strip()
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

        data = json.loads(cleaned)
        return CallAnalysis.model_validate(data)
