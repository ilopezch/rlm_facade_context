#!/usr/bin/env python3
"""
context_facade.py — Anthropic API facade for llama-server + RLM context compression
"""

import json
import logging
import re
import time
import uuid
import hashlib
from pathlib import Path
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from rlm import RLM
import tiktoken

# =============================================================================
# CONFIG
# =============================================================================
LLAMA_BASE_URL     = "http://localhost:8001"
SERVER_PORT        = 8081
SERVER_HOST        = "0.0.0.0"
LOG_LEVEL          = "INFO"
TOKEN_LIMIT        = 98_304
COMPRESS_THRESHOLD = 72_000
CHECKPOINT_DIR     = Path(".facade_anthropic_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("facade")

app = FastAPI(title="Claude Facade", version="2.0.0")
enc = tiktoken.get_encoding("cl100k_base")

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name":  "local",
        "base_url":    LLAMA_BASE_URL + "/v1",
        "api_key":     "local",
        "max_tokens":  16384,
    },
)


# =============================================================================
# Token counting + checkpointing
# =============================================================================
def count_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(b.get("text","") for b in content if isinstance(b,dict))
        total += len(enc.encode(str(content))) + 4
    return total


def save_checkpoint(messages: list[dict]) -> None:
    cid = hashlib.md5(json.dumps(messages[:3], sort_keys=True).encode()).hexdigest()[:12]
    (CHECKPOINT_DIR / f"{cid}.json").write_text(
        json.dumps({"saved_at": time.time(), "messages": messages}, indent=2)
    )


def compress_via_rlm(messages: list[dict]) -> list[dict]:
    TAIL = 6
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system  = [m for m in messages if m.get("role") != "system"]
    tail        = non_system[-TAIL:]
    to_compress = non_system[:-TAIL]
    if not to_compress:
        return messages

    history = "\n".join(
        f"[{m['role'].upper()}]: {m.get('content','')}" for m in to_compress
    )
    summary = rlm.completion(
        "Summarise this conversation preserving all file names, class names, "
        f"decisions and technical details:\n\n{history}"
    ).response.strip()

    reduced = system_msgs + [
        {"role": "assistant", "content": f"[CONTEXT SUMMARY]\n{summary}\n[END SUMMARY]"}
    ] + tail
    log.info("RLM compressed %d msgs | %d → %d tokens",
             len(to_compress), count_tokens(messages), count_tokens(reduced))
    return reduced


# =============================================================================
# Format helpers (from claude-local-server.py, adapted for llama-server)
# =============================================================================
# Tools Claude Code actually needs for /init
ESSENTIAL_TOOLS = {"Bash", "Read", "Write", "Edit"}

def build_system_prompt(body: dict) -> str:
    system = body.get("system", "")
    if isinstance(system, list):
        system = "\n".join(
            b.get("text","") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )

    tools = [t for t in body.get("tools", []) if t.get("name") in ESSENTIAL_TOOLS]
    if tools:
        tool_lines = "\n".join(
            f'- {t["name"]}: {t.get("description","")[:120]}'
            for t in tools
        )
        system += (
            "\n\n## IMPORTANT: You must use tools to complete tasks. "
            "Never describe what you would do — actually do it using tools.\n\n"
            "Available tools:\n"
            f"{tool_lines}\n\n"
            "To call a tool respond with ONLY this format — nothing else:\n"
            "<tool_call>\n"
            "<n>tool_name</n>\n"
            "<parameters>{\"param_name\": \"param_value\"}</parameters>\n"
            "</tool_call>\n\n"
            "Example for Read tool:\n"
            "<tool_call>\n"
            "<n>Read</n>\n"
            "<parameters>{\"file_path\": \"/path/to/file\"}</parameters>\n"
            "</tool_call>\n\n"
            "Example for Bash tool:\n"
            "<tool_call>\n"
            "<n>Bash</n>\n"
            "<parameters>{\"command\": \"ls -la\"}</parameters>\n"
            "</tool_call>"
        )
    return system

def anthropic_to_openai_messages(body: dict) -> list[dict]:
    messages = []
    system = build_system_prompt(body)
    if system:
        messages.append({"role": "system", "content": system})

    for m in body.get("messages", []):
        role    = m["role"]
        content = m.get("content", "")

        if isinstance(content, list):
            text_parts   = []
            tool_results = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                elif block.get("type") == "text":
                    text_parts.append(block.get("text",""))
                elif block.get("type") == "tool_result":
                    tc = block.get("content","")
                    if isinstance(tc, list):
                        tc = "\n".join(b.get("text","") for b in tc if isinstance(b,dict))
                    tool_results.append(f"[Tool result for {block.get('tool_use_id','')}]\n{tc}")
                elif block.get("type") == "tool_use":
                    # Assistant tool_use block — render as XML for context
                    inp = json.dumps(block.get("input",{}), ensure_ascii=False)
                    text_parts.append(
                        f"<tool_call>\n<n>{block.get('name','')}</n>\n"
                        f"<parameters>{inp}</parameters>\n</tool_call>"
                    )
            content = "\n".join(text_parts + tool_results)

        messages.append({"role": role, "content": content})
    return messages

def parse_tool_call(text: str) -> dict | None:
    # Primary: correct format
    pattern = r"<tool_call>\s*<n>(.*?)</n>\s*<parameters>(.*?)</parameters>\s*</tool_call>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        name = match.group(1).strip()
        raw  = match.group(2).strip()
        log.info("parse_tool_call name=%s raw_params=%.200s", name, raw)  # ADD THIS
        try:
            params = json.loads(match.group(2).strip())
        except json.JSONDecodeError:
            params = {"raw": match.group(2).strip()}
        return {"name": name, "input": params}

    # Fallback: <parameter name="ToolName">key="value"}</parameter>
    pattern2 = r'<parameter name="(\w+)">(.*?)</parameter>'
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        name = match2.group(1).strip()
        raw  = match2.group(2).strip()
        # Try to extract key="value" pairs
        try:
            # Fix malformed JSON: key="value"} → {"key": "value"}
            fixed = re.sub(r'(\w+)="([^"]*)"', r'"\1": "\2"', raw)
            fixed = re.sub(r'(\w+)=([^",}\s]+)', r'"\1": "\2"', fixed)
            if not fixed.strip().startswith("{"):
                fixed = "{" + fixed
            if not fixed.strip().endswith("}"):
                fixed = fixed + "}"
            params = json.loads(fixed)
        except Exception:
            params = {"raw": raw}
        return {"name": name, "input": params}

    return None

def build_anthropic_response(oai_response: dict, requested_model: str) -> dict:
    choice       = oai_response.get("choices", [{}])[0]
    message      = choice.get("message", {})
    content_text = message.get("content", "") or ""
    oai_tools    = message.get("tool_calls", [])

    # Strip <think> blocks
    content_text = re.sub(r"<think>.*?</think>", "", content_text, flags=re.DOTALL).strip()

    content_blocks = []
    stop_reason    = "end_turn"

    # Native tool_calls take priority over text parsing
    if oai_tools:
        for tc in oai_tools:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            content_blocks.append({
                "type":  "tool_use",
                "id":    tc.get("id", f"toolu_{uuid.uuid4().hex[:16]}"),
                "name":  fn.get("name", ""),
                "input": arguments,
            })
        stop_reason = "tool_use"
    else:
        # Fallback: parse <tool_call> from text
        tool_call = parse_tool_call(content_text)
        if tool_call:
            content_blocks.append({
                "type":  "tool_use",
                "id":    f"toolu_{uuid.uuid4().hex[:16]}",
                "name":  tool_call["name"],
                "input": tool_call["input"],
            })
            stop_reason = "tool_use"
        elif content_text:
            content_blocks.append({"type": "text", "text": content_text})

    if choice.get("finish_reason") == "length":
        stop_reason = "max_tokens"

    usage = oai_response.get("usage", {})
    return {
        "id":            f"msg_{uuid.uuid4().hex}",
        "type":          "message",
        "role":          "assistant",
        "content":       content_blocks,
        "model":         requested_model,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }

async def stream_anthropic(
    messages: list[dict],
    requested_model: str,
    temperature: float,
    max_tokens: int,
    anthropic_tools: list = [],
) -> AsyncIterator[bytes]:

    msg_id = f"msg_{uuid.uuid4().hex}"

    # Preamble — must arrive before any content
    preamble = [
        {"type": "message_start", "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": requested_model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
    ]
    for event in preamble:
        yield f"data: {json.dumps(event)}\n\n".encode()

    payload = {
        "model":       "local",
        "messages":    messages,
        "stream":      True,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    if anthropic_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {}),
                }
            }
            for t in anthropic_tools
        ]
        payload["tool_choice"] = "auto"

    full_text     = ""
    output_tokens = 0
    stop_reason   = "end_turn"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", f"{LLAMA_BASE_URL}/v1/chat/completions", json=payload
            ) as resp:

                if resp.status_code != 200:
                    error_body = await resp.aread()
                    log.error("llama-server stream error: %s %s", resp.status_code, error_body)
                    yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':f'upstream error {resp.status_code}'}})}\n\n".encode()
                    return

                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            parsed  = json.loads(data)
                            choice  = parsed.get("choices", [{}])[0]
                            delta   = choice.get("delta", {})
                            text    = delta.get("content") or ""
                            finish  = choice.get("finish_reason")

                            # Native tool_calls in stream
                            tool_calls = delta.get("tool_calls", [])
                            if tool_calls:
                                stop_reason = "tool_use"

                            if text:
                                full_text += text
                                clean = re.sub(
                                    r"<think>.*?</think>", "", text, flags=re.DOTALL
                                )
                                if clean:
                                    yield f"data: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':clean}})}\n\n".encode()

                            usage = parsed.get("usage") or {}
                            if usage.get("completion_tokens"):
                                output_tokens = usage["completion_tokens"]

                            if finish == "stop":
                                break
                            elif finish == "tool_calls":
                                stop_reason = "tool_use"
                                break

                        except json.JSONDecodeError:
                            continue

    except Exception as e:
        log.error("stream_anthropic error: %s", e)
        yield f"data: {json.dumps({'type':'error','error':{'type':'api_error','message':str(e)}})}\n\n".encode()
        return

    # Check for tool calls in full text if not already detected
    if stop_reason == "end_turn":
        clean_full = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
        if parse_tool_call(clean_full):
            stop_reason = "tool_use"

    yield f"data: {json.dumps({'type':'content_block_stop','index':0})}\n\n".encode()
    yield f"data: {json.dumps({'type':'message_delta','delta':{'stop_reason':stop_reason,'stop_sequence':None},'usage':{'output_tokens':output_tokens}})}\n\n".encode()
    yield b'data: {"type":"message_stop"}\n\n'

# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
@app.head("/")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "claude-sonnet-4.6", "object": "model", "created": int(time.time()), "owned_by": "local"},
    ]}


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    total_chars = len(str(body.get("system","")))
    for m in body.get("messages",[]):
        total_chars += len(str(m.get("content","")))
    return JSONResponse({"input_tokens": max(1, total_chars // 4)})


@app.post("/v1/messages")
async def create_message(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    requested_model = body.get("model", "claude-sonnet-4.6")
    stream          = body.get("stream", False)
    temperature     = body.get("temperature", 0.2)
    max_tokens      = body.get("max_tokens", 16384)
    anthropic_tools = body.get("tools", [])

    log.info("tools in request: %s", [t.get("name") for t in anthropic_tools])

    messages = anthropic_to_openai_messages(body)
    tokens   = count_tokens(messages)
    save_checkpoint(messages)

    log.info("→ stream=%s msgs=%d tokens=%d", stream, len(messages), tokens)

    if tokens > COMPRESS_THRESHOLD:
        log.info("Compressing via RLM (%d tokens)...", tokens)
        messages = compress_via_rlm(messages)

    # Build payload with native OpenAI function calling
    payload = {
        "model":       "local",
        "messages":    messages,
        "stream":      False,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    # Pass tools as native OpenAI functions — llama-server handles formatting
    if anthropic_tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name":        t["name"],
                    "description": t.get("description", ""),
                    "parameters":  t.get("input_schema", {}),
                }
            }
            for t in anthropic_tools
        ]
        payload["tool_choice"] = "auto"

    if stream:
        return StreamingResponse(
            stream_anthropic(messages, requested_model, temperature, max_tokens, anthropic_tools),
            media_type="text/event-stream",
            headers={
                "Cache-Control":   "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection":      "keep-alive",
                "Transfer-Encoding": "chunked",
            }
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            resp = await client.post(f"{LLAMA_BASE_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=str(e))

    anthropic_resp = build_anthropic_response(resp.json(), requested_model)
    log.info("← stop_reason=%s in=%d out=%d response_text=%.200s",
             anthropic_resp["stop_reason"],
             anthropic_resp["usage"]["input_tokens"],
             anthropic_resp["usage"]["output_tokens"],
             anthropic_resp["content"][0].get("text","") if anthropic_resp["content"] else "")
    return JSONResponse(anthropic_resp)

@app.post("/v1/chat/completions")
async def chat_completions_passthrough(request: Request):
    """Accept OpenAI-format requests and forward directly to llama-server."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    messages = body.get("messages", [])
    tokens   = count_tokens(messages)
    log.info("→ /v1/chat/completions stream=%s msgs=%d tokens=%d",
             body.get("stream"), len(messages), tokens)

    save_checkpoint(messages)

    if tokens > COMPRESS_THRESHOLD:
        body["messages"] = compress_via_rlm(messages)

    target = f"{LLAMA_BASE_URL}/v1/chat/completions"

    if body.get("stream"):
        return StreamingResponse(
            _passthrough_stream(target, body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform",
                     "X-Accel-Buffering": "no"},
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(target, json=body)
        return JSONResponse(resp.json(), status_code=resp.status_code)


async def _passthrough_stream(url: str, body: dict) -> AsyncIterator[bytes]:
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("POST", url, json=body) as resp:
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        yield (line.strip() + "\n\n").encode()

# =============================================================================
if __name__ == "__main__":
    log.info("Facade → llama-server: %s", LLAMA_BASE_URL)
    log.info("Listening on port %d", SERVER_PORT)
    log.info("Launch Claude Code with:")
    log.info("  ANTHROPIC_BASE_URL=http://localhost:%d ANTHROPIC_API_KEY=sk-local claude", SERVER_PORT)
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level=LOG_LEVEL.lower())
