from anthropic import AsyncAnthropic

from app.config import settings
from app.llm.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """LLM provider using Anthropic's API."""

    def __init__(self) -> None:
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.anthropic_model

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.1,
        )
        return response.content[0].text
