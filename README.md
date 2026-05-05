# rlm_facade_context
Anthropic API facade for llama-server + RLM context compression.

An Anthropic API facade that proxies requests to a locally-running `llama-server` (OpenAI-compatible). It translates between Anthropic's message format and OpenAI's format, and applies RLM (Recurrent Language Model) context compression when conversation history approaches token limits.

Typical usage: run this facade so that Claude Code (or any Anthropic API client) can use a local LLM instead of Anthropic's cloud.

