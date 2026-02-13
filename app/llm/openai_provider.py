import copy
import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.llm.base import BaseLLMProvider
from app.models import CallAnalysis

logger = logging.getLogger(__name__)


def _make_openai_strict_schema(pydantic_schema: dict) -> dict:
    """Transform a Pydantic JSON schema into OpenAI strict-mode format.

    OpenAI strict mode is very restrictive about what JSON Schema features
    it supports. Key constraints:
    1. ALL properties must be listed in "required" (even optional/nullable ones)
    2. "additionalProperties": false on every object
    3. No "$ref" allowed — all references must be inlined (resolved)
    4. No "allOf", "oneOf" at the property level
    5. No "minimum"/"maximum" on numbers — only "enum" constraints

    Our approach: resolve all $ref by inlining definitions, then fix up
    the schema recursively.
    """
    schema = copy.deepcopy(pydantic_schema)
    defs = schema.pop("$defs", {})

    # Recursively resolve $ref and fix nodes
    _resolve_and_fix(schema, defs)
    return schema


def _resolve_and_fix(node: dict, defs: dict) -> None:
    """Recursively resolve $ref references and apply strict-mode fixes."""
    # Fix object properties
    if "properties" in node:
        node["required"] = list(node["properties"].keys())
        node["additionalProperties"] = False

        for prop_name in list(node["properties"]):
            prop = node["properties"][prop_name]
            # Inline $ref — replace with the actual definition
            if "$ref" in prop:
                ref_name = prop["$ref"].split("/")[-1]  # "#/$defs/Foo" -> "Foo"
                resolved = copy.deepcopy(defs[ref_name])
                # Keep sibling keys like "description" from the property
                desc = prop.get("description")
                node["properties"][prop_name] = resolved
                if desc:
                    node["properties"][prop_name]["description"] = desc
                prop = node["properties"][prop_name]

            # Remove min/max constraints (not allowed in strict mode)
            prop.pop("minimum", None)
            prop.pop("maximum", None)

            # Recurse into nested structures
            _resolve_and_fix(prop, defs)

            # Also resolve $refs inside anyOf (for Optional fields)
            if "anyOf" in prop:
                for i, option in enumerate(prop["anyOf"]):
                    if "$ref" in option:
                        ref_name = option["$ref"].split("/")[-1]
                        prop["anyOf"][i] = copy.deepcopy(defs[ref_name])


# Build the response format once at import time
_CALL_ANALYSIS_SCHEMA = _make_openai_strict_schema(
    CallAnalysis.model_json_schema()
)

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "call_analysis",
        "strict": True,
        "schema": _CALL_ANALYSIS_SCHEMA,
    },
}


class OpenAIProvider(BaseLLMProvider):
    """LLM provider using OpenAI's API."""

    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=_RESPONSE_FORMAT,
            temperature=0.1,
        )
        # Log token usage — OpenAI returns this in every response.
        # Useful for cost tracking: input tokens are cheap, output tokens cost more.
        if response.usage:
            logger.info(
                "OpenAI token usage",
                extra={"extra_data": {
                    "model": self.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }},
            )
        return response.choices[0].message.content or ""

    async def _stream_llm(
        self, system_prompt: str, user_message: str
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=_RESPONSE_FORMAT,
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
