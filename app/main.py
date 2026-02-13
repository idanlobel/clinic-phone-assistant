import argparse
import asyncio
import json
import logging
import sys
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.logging_config import setup_logging
from app.llm.base import BaseLLMProvider
from app.models import CallAnalysis, TranscriptRequest
from app.security import check_rate_limit, verify_api_key

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinic Phone Assistant",
    description="AI-powered clinic phone call transcript analyzer",
    version="1.0.0",
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Middleware that wraps every HTTP request with logging and tracing.

    This runs BEFORE and AFTER every endpoint, adding:
    - A unique request_id for tracing through logs
    - Request timing (how long the endpoint took)
    - HTTP method, path, and status code

    In production, you'd correlate all logs from a single request
    using the request_id.
    """
    request_id = str(uuid.uuid4())[:8]  # Short ID for readability
    start_time = time.perf_counter()

    # Store request_id so endpoints can access it
    request.state.request_id = request_id

    logger.info(
        "Request started",
        extra={"extra_data": {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        }},
    )

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "Request completed",
        extra={"extra_data": {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }},
    )

    # Add request_id to response headers so clients can reference it
    response.headers["X-Request-ID"] = request_id
    return response


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
async def analyze_transcript(
    request: TranscriptRequest,
    api_key: str = Depends(verify_api_key),  # Auth check runs first
) -> CallAnalysis:
    """Analyze a phone call transcript and return structured information."""
    # Rate limit check (uses the authenticated API key as identifier)
    await check_rate_limit(api_key)

    provider = get_provider()
    try:
        result = await provider.analyze(request.transcript)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM analysis failed: {e}")
    return result


@app.post("/analyze/stream")
async def analyze_transcript_stream(
    request: TranscriptRequest,
    api_key: str = Depends(verify_api_key),
) -> StreamingResponse:
    """Stream the analysis result token-by-token using Server-Sent Events (SSE)."""
    await check_rate_limit(api_key)
    provider = get_provider()

    async def event_stream():
        """Async generator that formats LLM chunks as SSE events."""
        async for chunk in provider.stream_analyze(request.transcript):
            # SSE format: "data: <content>\n\n"
            # Each chunk is a few tokens of the JSON response
            yield f"data: {json.dumps(chunk)}\n\n"
        # Send a final event to signal completion
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",  # Tells the browser this is SSE
        headers={
            "Cache-Control": "no-cache",       # Don't cache the stream
            "Connection": "keep-alive",        # Keep the HTTP connection open
            "X-Accel-Buffering": "no",         # Disable nginx buffering if behind a proxy
        },
    )


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
