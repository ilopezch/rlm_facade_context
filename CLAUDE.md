# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

An Anthropic API facade that proxies requests to a locally-running `llama-server` (OpenAI-compatible). It translates between Anthropic's message format and OpenAI's format, and applies RLM (Recurrent Language Model) context compression when conversation history approaches token limits.

Typical usage: run this facade so that Claude Code (or any Anthropic API client) can use a local LLM instead of Anthropic's cloud.

```bash
ANTHROPIC_BASE_URL=http://localhost:8081 ANTHROPIC_API_KEY=sk-local claude
```

## Running

**Prerequisites:** `llama-server` running on `http://localhost:8001`

```bash
pip install -r requirements.txt
python rlm_facade_context.py
```

The server listens on port `8081` by default.

## Architecture

Single-file application: `rlm_facade_context.py`. Key configuration constants are at the top of the file:

| Constant | Default | Purpose |
|----------|---------|---------|
| `LLAMA_BASE_URL` | `http://localhost:8001` | Upstream llama-server |
| `SERVER_PORT` | `8081` | Facade listen port |
| `TOKEN_LIMIT` | `98304` | Hard token ceiling |
| `COMPRESS_THRESHOLD` | `72000` | Triggers RLM compression |
| `CHECKPOINT_DIR` | `.facade_anthropic_checkpoints` | Conversation snapshots |

**Request flow:**
1. Client sends Anthropic-format request to `POST /v1/messages`
2. Messages are converted from Anthropic → OpenAI format (`anthropic_to_openai_messages`)
3. Tokens are counted via tiktoken (`cl100k_base`); a checkpoint is saved
4. If token count exceeds `COMPRESS_THRESHOLD`, older messages are compressed by RLM into a summary block (keeps last 6 messages + all system prompts)
5. Request is forwarded to llama-server; response is converted back to Anthropic format (`build_anthropic_response`)
6. Streaming responses use SSE via `stream_anthropic()`

**API endpoints:**
- `GET /` — health check
- `GET /v1/models` — returns `claude-sonnet-4.6`
- `POST /v1/messages` — main Anthropic-compatible endpoint (streaming supported)
- `POST /v1/messages/count_tokens` — token counting
- `POST /v1/chat/completions` — OpenAI format passthrough

**Tool calling:** Supports native OpenAI tool call format plus a fallback XML-based parser (`parse_tool_call`) for models that emit tools as text.

## Dependencies

- `fastapi` + `uvicorn` — web framework/server
- `httpx` — async HTTP client for proxying to llama-server
- `tiktoken` — token counting (cl100k_base encoding)
- `rlm` (from GitHub: `alexzhang13/rlm`) — context compression; configured with an OpenAI-compatible backend pointed at the local llama-server
