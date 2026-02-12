# Clinic Phone Assistant

An AI-powered clinic phone call assistant that analyzes call transcripts to extract structured patient information, classify caller intent, and flag urgent medical issues.

## Features

- **Intent Classification** — Identifies the caller's purpose (appointment booking, prescription refill, billing, urgent medical issue, etc.)
- **Entity Extraction** — Pulls out name, date of birth, callback number, and reason for call
- **Urgency Flagging** — Automatically flags high-urgency calls involving dangerous symptoms (chest pain, breathing difficulty, etc.)
- **Dual LLM Support** — Works with both OpenAI (GPT-4o) and Anthropic (Claude), configurable via environment variable
- **REST API + CLI** — Use as a web service or directly from the command line

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env with your API key(s)
```

Set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` depending on which provider you want to use.

### 3. Run via CLI

```bash
python -m app.main --transcript "Hi, this is Sarah Cohen, born 03/12/1988. I need to book an appointment because I've had chest pain for two days. Please call me back at 310-555-2211."
```

Output:

```json
{
  "intent": "urgent_medical_issue",
  "name": "Sarah Cohen",
  "dob": "1988-03-12",
  "phone": "310-555-2211",
  "summary": "Chest pain for two days, needs an appointment",
  "urgency": "high",
  "confidence": 0.95
}
```

### 4. Run as a web server

```bash
python -m app.main --serve
```

Then send requests:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Hi, this is Sarah Cohen, born 03/12/1988. I need to book an appointment because I have had chest pain for two days. Please call me back at 310-555-2211."}'
```

API docs are available at `http://localhost:8000/docs` (Swagger UI).

### 5. Run tests

```bash
pytest tests/ -v
```

## Project Structure

```
app/
  main.py          — FastAPI app + CLI entrypoint
  models.py        — Pydantic models (request/response schemas)
  prompt.py        — LLM system prompt with few-shot examples
  config.py        — Settings loaded from .env
  llm/
    base.py        — Abstract LLM provider interface + JSON parsing
    openai_provider.py
    anthropic_provider.py
tests/
  test_assistant.py — Unit tests (models, parsing, API endpoints)
```

## How I Approached the Problem

1. **Started with the data model** — Defined the exact input/output contract using Pydantic, ensuring the schema covers all required fields plus a confidence score.

2. **Focused on prompt engineering** — Wrote a detailed system prompt with clear classification rules, urgency guidelines (chest pain = always high), date format handling, and few-shot examples. This is where the "intelligence" lives.

3. **Clean provider abstraction** — Both OpenAI and Anthropic are behind a simple interface (`BaseLLMProvider`), making it trivial to swap providers or add new ones. The parsing logic (handling markdown fences, JSON validation) is shared.

4. **Dual interface** — Both CLI and REST API share the same analysis logic. The CLI is useful for quick testing; the API is what a real production system would expose.

5. **Test without API calls** — All tests use mocked LLM responses so they run fast, offline, and deterministically.

## AI Tools Used

- **Claude Code (Anthropic CLI)** — Used to scaffold the entire project, write all source code, design the architecture, and author this README. I used it as a pair-programming partner: I made the high-level design decisions (tech stack, architecture, provider abstraction) and Claude handled the implementation details.

## What I Would Improve With More Time

- **Streaming responses** — For longer transcripts, stream the LLM response to reduce perceived latency
- **Multi-turn conversation support** — Handle transcripts with multiple speakers (e.g., IVR menu navigation, receptionist interactions)
- **Structured output mode** — Use OpenAI's native structured output feature and Anthropic's tool use for guaranteed JSON schema compliance instead of parsing raw text
- **Logging & observability** — Add structured logging, request tracing, and LLM token usage tracking
- **Confidence calibration** — Use multiple LLM calls or chain-of-thought prompting to improve confidence scoring accuracy
- **Input validation** — Detect and reject non-transcript inputs (random text, code, etc.)
- **Rate limiting & auth** — Add API key authentication and rate limiting for the REST endpoint
- **Containerization** — Add a Dockerfile for easy deployment
