import json
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings
from app.llm.base import BaseLLMProvider
from app.models import CallAnalysis

# Define a "tool" whose input schema matches CallAnalysis.
# When we tell Claude to use this tool, it must return valid JSON
# matching our schema — Anthropic validates this server-side.
_ANALYSIS_TOOL = {
    "name": "submit_analysis",
    "description": (
        "Submit the structured analysis of a phone call transcript. "
        "You MUST use this tool to respond."
    ),
    "input_schema": CallAnalysis.model_json_schema(),
}


class AnthropicProvider(BaseLLMProvider):
    """LLM provider using Anthropic's API."""

    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        # tool_choice="any" forces Claude to use one of the provided tools,
        # guaranteeing structured output instead of free-form text.
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[_ANALYSIS_TOOL],
            tool_choice={"type": "any"},
            temperature=0.1,
        )
        # The response contains a tool_use block with our structured data.
        # We find it and return the JSON string.
        for block in response.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
        return response.content[0].text

    async def _stream_llm(
        self, system_prompt: str, user_message: str
    ) -> AsyncIterator[str]:
        # For streaming with tool use, Anthropic streams the tool input JSON.
        # We use the standard text_stream which includes tool input deltas.
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=[_ANALYSIS_TOOL],
            tool_choice={"type": "any"},
            temperature=0.1,
        ) as stream:
            async for text in stream.text_stream:
                yield text
