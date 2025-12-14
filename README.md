# Open WebUI Anthropic Pipe with Agentic Tool Execution

A function pipe for Open WebUI that integrates Anthropic Claude models with full tool execution support via MCP (Model Context Protocol).

## Features

- All Claude 3, 3.5, 3.7, 4, 4.5, Opus 4.1, and Opus 4.5 models
- Streaming responses
- Image and PDF processing
- **Prompt caching** - automatic caching of tools and system prompts (90% token savings)
- **Token-efficient tools** - optimized tool token usage
- Extended thinking (Claude 3.7+)
- **Agentic tool loop** - tools executed in-pipe, not just returned as JSON

## Prompt Caching

The pipe automatically enables prompt caching for significant token savings:

| Cached Content | Trigger | Savings |
|----------------|---------|---------|
| Tools (25+ definitions) | Always when tools present | 90% on repeat calls |
| System prompt | When >4000 chars | 90% on repeat calls |

Beta headers automatically added:
- `prompt-caching-2024-07-31` - enables caching
- `token-efficient-tools-2025-02-19` - optimizes tool tokens

First call pays +25% (cache write), subsequent calls pay only 10% (cache read).

## Agentic Loop Architecture

Open WebUI is a chat interface, not designed for CLI-style agentic loops. This pipe solves that:

1. Non-streaming first call to get tool_use blocks
2. Execute tools via Desktop Commander MCP server
3. Stream the interpretation back to the user

All yields happen in ONE generator that Open WebUI follows from start to finish.

**Status events:**
- "Thinking..." - waiting for initial response
- "Executing tools..." - running MCP tools
- "Interpreting..." - streaming final response
- "Done" - complete

## Requirements

- Open WebUI
- Anthropic API key
- Desktop Commander MCP server running via MCPO (optional, for tool execution)

## Installation

1. In Open WebUI, go to Admin Panel > Functions
2. Create new function, paste the contents of `anthropic_pipe.py`
3. Configure the valves:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key
   - `DESKTOP_COMMANDER_URL`: URL to your Desktop Commander MCP server (default: `http://localhost:8000/desktop-commander`)
   - `EXECUTE_TOOLS_IN_PIPE`: Set to `true` to execute tools directly

## Configuration

Key valves:

| Valve | Default | Description |
|-------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `ENABLE_THINKING` | false | Enable extended thinking for Claude 3.7+ |
| `EXECUTE_TOOLS_IN_PIPE` | true | Execute MCP tools directly in the pipe |
| `DESKTOP_COMMANDER_URL` | localhost:8000 | URL to Desktop Commander MCP server |

## Tool Execution

When `EXECUTE_TOOLS_IN_PIPE` is enabled, the pipe:

1. Converts OpenAI-format tools to Anthropic format
2. Makes a non-streaming call to detect tool_use
3. Executes tools via Desktop Commander HTTP API
4. Sends tool results back to Claude
5. Streams the interpretation

Tool name mapping: `tool_read_file_post` maps to endpoint `/read_file`

## Credits

- Original pipe by Balaxxe
- Agentic loop, caching, and MCP integration by Radar105 & Aurora

## License

MIT
