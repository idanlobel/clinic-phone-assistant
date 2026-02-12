import argparse
import asyncio
import json
import sys

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.llm.base import BaseLLMProvider
from app.models import CallAnalysis, TranscriptRequest

app = FastAPI(
    title="Clinic Phone Assistant",
    description="AI-powered clinic phone call transcript analyzer",
    version="1.0.0",
)


def get_provider() -> BaseLLMProvider:
    """Return the configured LLM provider instance."""
    if settings.llm_provider == "openai":
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider()
    else:
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "provider": settings.llm_provider}


@app.post("/analyze", response_model=CallAnalysis)
async def analyze_transcript(request: TranscriptRequest) -> CallAnalysis:
    """Analyze a phone call transcript and return structured information."""
    provider = get_provider()
    try:
        result = await provider.analyze(request.transcript)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {e}")
    return result


async def cli_analyze(transcript: str) -> None:
    """Run analysis from the command line."""
    provider = get_provider()
    result = await provider.analyze(transcript)
    print(json.dumps(result.model_dump(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clinic Phone Assistant")
    parser.add_argument(
        "--transcript", "-t", type=str, help="Phone call transcript to analyze"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=True)
    elif args.transcript:
        asyncio.run(cli_analyze(args.transcript))
    else:
        # Read from stdin if no transcript provided
        print("Enter the phone call transcript (press Ctrl+D / Ctrl+Z when done):")
        transcript = sys.stdin.read().strip()
        if transcript:
            asyncio.run(cli_analyze(transcript))
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
