# Open WebUI Anthropic Pipe

A function pipe for Open WebUI that integrates Anthropic Claude models with native tool execution support. Translates between Anthropic and OpenAI formats so Open WebUI v0.7's middleware handles the full tool loop.

## Features

- All Claude 3, 3.5, 3.7, 4, 4.5, Opus 4.1, Opus 4.5, and Opus 4.6 models
- Streaming responses with OpenAI-format `delta.tool_calls`
- Image and PDF processing
- Prompt caching (server-side)
- Extended thinking (Claude 3.7+)
- 128K output token support (Claude 3.7+)
- Native Open WebUI v0.7 tool execution — no in-pipe tool handling needed

## Architecture

The pipe is a **pure format translator** between Anthropic's API and OpenAI's streaming format. Open WebUI v0.7 handles everything else:

1. Open WebUI sends request to pipe (with tools in body)
2. Pipe converts OpenAI tools to Anthropic format, calls Anthropic API
3. Pipe streams response, yielding `delta.tool_calls` dicts for tool_use blocks
4. Open WebUI middleware parses tool_calls, executes tools via its tool_server connections
5. Middleware re-calls pipe with tool results as `role=tool` messages
6. Pipe converts tool results to Anthropic `tool_result` format, calls API again
7. Repeat until model stops requesting tools

The pipe never executes tools itself — it just translates formats.

## Requirements

- Open WebUI v0.7+
- Anthropic API key
- Tool server (e.g., MCPO) configured in Open WebUI for tool execution

## Installation

1. In Open WebUI, go to **Admin Panel > Functions**
2. Create new function, paste the contents of `anthropic_pipe.py`
3. Set the `ANTHROPIC_API_KEY` valve to your API key

## Configuration

| Valve | Default | Description |
|-------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `ENABLE_THINKING` | false | Enable extended thinking for Claude 3.7+ |
| `THINKING_BUDGET_TOKENS` | 16000 | Max tokens for thinking (when enabled) |
| `MAX_OUTPUT_TOKENS` | true | Use maximum output tokens for the model |
| `ENABLE_TOOL_CHOICE` | true | Pass tools to the Anthropic API |
| `ENABLE_SYSTEM_PROMPT` | true | Include system prompt in API calls |

## Supported Models

All Claude models from Claude 3 through Opus 4.6, including:

- Claude 3 (Opus, Sonnet, Haiku)
- Claude 3.5 (Sonnet, Haiku)
- Claude 3.7 Sonnet
- Claude Sonnet 4
- Claude Opus 4
- Claude Sonnet 4.5
- Claude Opus 4.1
- Claude Opus 4.5
- **Claude Opus 4.6** (latest, 128K output with beta header)

## Version History

- **v6.0.1** — Buffer and discard tool narration text (Claude emits `<parameter>` XML alongside tool_use blocks)
- **v6.0** — Removed in-pipe tool execution, aligned with Open WebUI v0.7 native tool handling
- **v5.1** — Added Claude Opus 4.6
- **v5.0** — Agentic loop for tool execution, Claude Opus 4.5, MCP tool integration
- **v4.0** — Added Claude 4, Claude 4.5 Sonnet, Opus 4.1
- **v3.0** — Added Claude 3.7, thinking valves, CoT streaming

## Credits

- Original pipe by [Balaxxe](https://openwebui.com/f/balaxxe/anthropic_manifold_pipe/)
- v5.0+ by Radar105
- [Desktop Commander MCP](https://github.com/wonderwhy-er/DesktopCommanderMCP) by wonderwhy-er (used in v5.0, removed in v6.0)

## License

MIT
