"""
title: Anthropic API Integration - Claude 4, 4.5, Opus 4.5 + Extended Thinking
author: Balaxxe / Aurora (Updated for Opus 4.5)
version: 5.0 - Aurora Network
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0
environment_variables:
    - ANTHROPIC_API_KEY (required)

Supports:
- All Claude 3, 3.5, 3.7, 4, 4.5, Opus 4.1, and Opus 4.5 models
- Streaming responses
- Image processing
- Prompt caching (server-side)
- Function calling
- PDF processing
- Cache Control
- Extended thinking (Claude 3.7+, 4+)

Updates:
v5.0 - Added Claude Opus 4.5 (claude-opus-4-5-20251101)
v4.0 - Added Claude 4, Claude 4.5 Sonnet, Opus 4.1
v3.0 - Added Claude 3.7, thinking valves, CoT streaming
"""

import os
import json
import time
import hashlib
import logging
import asyncio
from datetime import datetime
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    Optional,
    AsyncIterator,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import aiohttp


class Pipe:
    API_VERSION = "2023-06-01"
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    SUPPORTED_PDF_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-latest",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-0",
        "claude-opus-4-20250514",
        "claude-opus-4-0",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-5",
        "claude-opus-4-1-20250805",
        "claude-opus-4-1",
        "claude-opus-4-5-20251101",
        "claude-opus-4-5",
    ]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024
    PDF_BETA_HEADER = "pdfs-2024-09-25"
    OUTPUT_128K_BETA = "output-128k-2025-02-19"
    TOKEN_EFFICIENT_TOOLS = "token-efficient-tools-2025-02-19"
    # Model max tokens - updated with Claude 3.7, 4, 4.5, Opus 4.1 values
    MODEL_MAX_TOKENS = {
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-haiku-20241022": 8192,
        "claude-3-opus-latest": 4096,
        "claude-3-5-sonnet-latest": 8192,
        "claude-3-5-haiku-latest": 8192,
        "claude-3-7-sonnet-latest": 16384,  # Claude 3.7 supports up to 16K output tokens by default, 128K with beta
        # Claude 4 family
        "claude-sonnet-4-20250514": 16384,  # Claude Sonnet 4 - 16K default
        "claude-sonnet-4-0": 16384,  # Claude Sonnet 4 alias
        "claude-opus-4-20250514": 16384,  # Claude Opus 4 - 16K default
        "claude-opus-4-0": 16384,  # Claude Opus 4 alias
        "claude-sonnet-4-5-20250929": 32768,  # Claude Sonnet 4.5 - 32K default
        "claude-sonnet-4-5": 32768,  # Claude Sonnet 4.5 alias
        "claude-opus-4-1-20250805": 32768,  # Claude Opus 4.1 - 32K default
        "claude-opus-4-1": 32768,  # Claude Opus 4.1 alias
        "claude-opus-4-5-20251101": 32768,  # Claude Opus 4.5 - 32K default
        "claude-opus-4-5": 32768,  # Claude Opus 4.5 alias
    }
    # Model context lengths - maximum input tokens
    MODEL_CONTEXT_LENGTH = {
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-opus-latest": 200000,
        "claude-3-5-sonnet-latest": 200000,
        "claude-3-5-haiku-latest": 200000,
        "claude-3-7-sonnet-latest": 200000,  # Claude 3.7 supports up to 200K context
        # Claude 4 family (1M context with beta header)
        "claude-sonnet-4-20250514": 200000,  # Claude Sonnet 4 - 200K default, 1M with beta
        "claude-sonnet-4-0": 200000,  # Claude Sonnet 4 alias
        "claude-opus-4-20250514": 200000,  # Claude Opus 4 - 200K default, 1M with beta
        "claude-opus-4-0": 200000,  # Claude Opus 4 alias
        "claude-sonnet-4-5-20250929": 200000,  # Claude Sonnet 4.5 - 200K default, 1M with beta
        "claude-sonnet-4-5": 200000,  # Claude Sonnet 4.5 alias
        "claude-opus-4-1-20250805": 200000,  # Claude Opus 4.1 - 200K default, 1M with beta
        "claude-opus-4-1": 200000,  # Claude Opus 4.1 alias
        "claude-opus-4-5-20251101": 200000,  # Claude Opus 4.5 - 200K default, 1M with beta
        "claude-opus-4-5": 200000,  # Claude Opus 4.5 alias
    }
    BETA_HEADER = "prompt-caching-2024-07-31"
    REQUEST_TIMEOUT = 120  # Increased timeout for longer responses
    THINKING_BUDGET_TOKENS = 16000  # Default thinking budget tokens

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = "Your API Key Here"
        ENABLE_THINKING: bool = (
            False  # Disabled - may interfere with tool calling
        )
        MAX_OUTPUT_TOKENS: bool = True  # Valve to use maximum possible output tokens
        ENABLE_TOOL_CHOICE: bool = True  # Valve to enable tool choice
        ENABLE_SYSTEM_PROMPT: bool = True  # Valve to enable system prompt
        THINKING_BUDGET_TOKENS: int = Field(
            default=16000, ge=0, le=16000
        )  # Configurable thinking budget tokens 16,000 max
        DESKTOP_COMMANDER_URL: str = "http://100.103.123.50:8000/desktop-commander"
        EXECUTE_TOOLS_IN_PIPE: bool = True  # Execute MCP tools directly in pipe

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None

    async def execute_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool via Desktop Commander MCP server"""
        # Map tool name to endpoint: tool_read_file_post -> /read_file
        if tool_name.startswith("tool_") and tool_name.endswith("_post"):
            endpoint = "/" + tool_name[5:-5]  # Remove "tool_" prefix and "_post" suffix
        else:
            endpoint = "/" + tool_name

        url = f"{self.valves.DESKTOP_COMMANDER_URL}{endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=arguments, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        result = await response.text()
                        # Try to parse as JSON and extract content
                        try:
                            parsed = json.loads(result)
                            if isinstance(parsed, str):
                                return parsed
                            return json.dumps(parsed, indent=2)
                        except:
                            return result
                    else:
                        return f"Error: HTTP {response.status} - {await response.text()}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def get_anthropic_models(self) -> List[dict]:
        # Models that support extended thinking
        thinking_models = [
            "claude-3-7-sonnet-latest",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-0",
            "claude-opus-4-20250514",
            "claude-opus-4-0",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4-5",
            "claude-opus-4-1-20250805",
            "claude-opus-4-1",
            "claude-opus-4-5-20251101",
            "claude-opus-4-5",
        ]

        return [
            {
                "id": f"anthropic/{name}",
                "name": name,
                "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                "supports_vision": name != "claude-3-5-haiku-20241022",
                "supports_thinking": name in thinking_models,
            }
            for name in [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
                "claude-3-7-sonnet-latest",
                "claude-sonnet-4-20250514",
                "claude-sonnet-4-0",
                "claude-opus-4-20250514",
                "claude-opus-4-0",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-5",
                "claude-opus-4-1-20250805",
                "claude-opus-4-1",
                "claude-opus-4-5-20251101",
                "claude-opus-4-5",
            ]
        ]

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_content(self, content: Union[str, List[dict]]) -> List[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed_content = []
        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                processed_content.append(self.process_image(item))
            elif item["type"] == "pdf_url":
                model_name = item.get("model", "").split("/")[-1]
                if model_name not in self.SUPPORTED_PDF_MODELS:
                    raise ValueError(
                        f"PDF support is only available for models: {', '.join(self.SUPPORTED_PDF_MODELS)}"
                    )
                processed_content.append(self.process_pdf(item))
            elif item["type"] == "tool_calls":
                processed_content.append(item)
            elif item["type"] == "tool_results":
                processed_content.append(item)
        return processed_content

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            if media_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported media type: {media_type}")

            # Check image size
            image_size = len(base64_data) * 3 / 4  # Approximate size of decoded base64
            if image_size > self.MAX_IMAGE_SIZE:
                raise ValueError(
                    f"Image size exceeds {self.MAX_IMAGE_SIZE/(1024*1024)}MB limit: {image_size/(1024*1024):.2f}MB"
                )

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["image_url"]["url"]},
            }

    def process_pdf(self, pdf_data):
        if pdf_data["pdf_url"]["url"].startswith("data:application/pdf"):
            mime_type, base64_data = pdf_data["pdf_url"]["url"].split(",", 1)

            # Check PDF size
            pdf_size = len(base64_data) * 3 / 4  # Approximate size of decoded base64
            if pdf_size > self.MAX_PDF_SIZE:
                raise ValueError(
                    f"PDF size exceeds {self.MAX_PDF_SIZE/(1024*1024)}MB limit: {pdf_size/(1024*1024):.2f}MB"
                )

            document = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_data,
                },
            }

            if pdf_data.get("cache_control"):
                document["cache_control"] = pdf_data["cache_control"]

            return document
        else:
            document = {
                "type": "document",
                "source": {"type": "url", "url": pdf_data["pdf_url"]["url"]},
            }

            if pdf_data.get("cache_control"):
                document["cache_control"] = pdf_data["cache_control"]

            return document

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, AsyncIterator[str]]:
        """
        Process a request to the Anthropic API.

        Args:
            body: The request body containing messages and parameters
            __event_emitter__: Optional event emitter for status updates

        Returns:
            Either a string response or an async iterator for streaming responses
        """
        if not self.valves.ANTHROPIC_API_KEY:
            error_msg = "Error: ANTHROPIC_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return {"content": error_msg, "format": "text"}

        try:
            system_message, messages = pop_system_message(body["messages"])

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Processing request...", "done": False},
                    }
                )

            model_name = body["model"].split("/")[-1]
            if model_name not in self.MODEL_MAX_TOKENS:
                logging.warning(
                    f"Unknown model: {model_name}, using default token limit"
                )

            # Get max tokens for the model
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_name, 4096)

            # If MAX_OUTPUT_TOKENS valve is enabled, use the maximum possible tokens for the model
            if self.valves.MAX_OUTPUT_TOKENS:
                max_tokens = max_tokens_limit
            else:
                max_tokens = min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                )

            payload = {
                "model": model_name,
                "messages": self._process_messages(messages),
                "max_tokens": max_tokens,
                "temperature": (
                    float(body.get("temperature"))
                    if body.get("temperature") is not None
                    else None
                ),
                "top_k": (
                    int(body.get("top_k")) if body.get("top_k") is not None else None
                ),
                "top_p": (
                    float(body.get("top_p")) if body.get("top_p") is not None else None
                ),
                "stream": body.get("stream", False),
                "metadata": body.get("metadata", {}),
            }

            # Add thinking parameter with proper format if enabled and model supports it
            thinking_models = [
                "claude-3-7-sonnet-latest",
                "claude-sonnet-4-20250514",
                "claude-sonnet-4-0",
                "claude-opus-4-20250514",
                "claude-opus-4-0",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-5",
                "claude-opus-4-1-20250805",
                "claude-opus-4-1",
                "claude-opus-4-5-20251101",
                "claude-opus-4-5",
            ]
            if self.valves.ENABLE_THINKING and model_name in thinking_models:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.valves.THINKING_BUDGET_TOKENS,
                }

            payload = {k: v for k, v in payload.items() if v is not None}

            # Add system message if enabled - with caching for long prompts
            if system_message and self.valves.ENABLE_SYSTEM_PROMPT:
                system_str = str(system_message)
                # Use array format with cache_control if system prompt is long enough (1024+ tokens ~= 4000+ chars)
                if len(system_str) > 4000:
                    payload["system"] = [
                        {
                            "type": "text",
                            "text": system_str,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                else:
                    payload["system"] = system_str

            # Add tools if enabled - convert OpenAI format to Anthropic format
            if "tools" in body and self.valves.ENABLE_TOOL_CHOICE:
                anthropic_tools = []
                for tool in body["tools"]:
                    # Handle OpenAI format: {"type": "function", "function": {...}}
                    if isinstance(tool, dict) and tool.get("type") == "function":
                        func = tool.get("function", {})
                        anthropic_tools.append({
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                        })
                    # Handle raw function format: {"name": ..., "description": ..., "parameters": ...}
                    elif isinstance(tool, dict) and "name" in tool:
                        anthropic_tools.append({
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "input_schema": tool.get("parameters", {"type": "object", "properties": {}})
                        })
                if anthropic_tools:
                    # Add cache_control to the final tool (Anthropic caching requirement)
                    if len(anthropic_tools) > 0:
                        anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
                    payload["tools"] = anthropic_tools
                # Convert OpenAI tool_choice format to Anthropic format
                tool_choice = body.get("tool_choice")
                if tool_choice:
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            payload["tool_choice"] = {"type": "auto"}
                        elif tool_choice == "required" or tool_choice == "any":
                            payload["tool_choice"] = {"type": "any"}
                        # "none" means don't use tools, so we skip adding tool_choice
                    elif isinstance(tool_choice, dict):
                        payload["tool_choice"] = tool_choice
                else:
                    # Default to auto when tools are present
                    payload["tool_choice"] = {"type": "auto"}

            if "response_format" in body:
                payload["response_format"] = {
                    "type": body["response_format"].get("type")
                }

            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }

            beta_headers = []

            # Add PDF beta header if needed
            if any(
                isinstance(msg["content"], list)
                and any(item.get("type") == "pdf_url" for item in msg["content"])
                for msg in body.get("messages", [])
            ):
                beta_headers.append(self.PDF_BETA_HEADER)

            # Add cache control beta header if needed (check messages AND system prompt)
            has_message_cache = any(
                isinstance(msg["content"], list)
                and any(item.get("cache_control") for item in msg["content"])
                for msg in body.get("messages", [])
            )
            has_system_cache = (
                isinstance(payload.get("system"), list)
                and any(item.get("cache_control") for item in payload.get("system", []))
            )
            # Check if tools have cache_control
            has_tools_cache = (
                isinstance(payload.get("tools"), list)
                and len(payload.get("tools", [])) > 0
                and payload.get("tools", [{}])[-1].get("cache_control")
            )
            if has_message_cache or has_system_cache or has_tools_cache:
                beta_headers.append(self.BETA_HEADER)

            # Add token-efficient-tools beta header when tools are present
            if "tools" in payload and len(payload.get("tools", [])) > 0:
                beta_headers.append(self.TOKEN_EFFICIENT_TOOLS)

            # Add 128K output beta header for Claude 3.7+ and 4+ models
            extended_output_models = [
                "claude-3-7-sonnet-latest",
                "claude-sonnet-4-20250514",
                "claude-sonnet-4-0",
                "claude-opus-4-20250514",
                "claude-opus-4-0",
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-5",
                "claude-opus-4-1-20250805",
                "claude-opus-4-1",
                "claude-opus-4-5-20251101",
                "claude-opus-4-5",
            ]
            if model_name in extended_output_models and self.valves.MAX_OUTPUT_TOKENS:
                beta_headers.append(self.OUTPUT_128K_BETA)

            if beta_headers:
                headers["anthropic-beta"] = ",".join(beta_headers)

            try:
                # If tools are present and we want to execute them, use non-streaming first
                # to get the complete response, execute tools, then stream the interpretation
                has_tools = "tools" in payload and self.valves.EXECUTE_TOOLS_IN_PIPE

                if payload["stream"] and not has_tools:
                    # No tools - just stream normally
                    return self._stream_with_ui(
                        self.MODEL_URL, headers, payload, body, __event_emitter__
                    )

                if has_tools:
                    # Tools present - do agentic loop
                    return self._agentic_tool_loop(
                        self.MODEL_URL, headers, payload, body, __event_emitter__
                    )

                response_data, cache_metrics = await self._send_request(
                    self.MODEL_URL, headers, payload
                )

                if (
                    isinstance(response_data, dict)
                    and "content" in response_data
                    and response_data.get("format") == "text"
                ):
                    # This is an error response
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": response_data["content"],
                                    "done": True,
                                },
                            }
                        )
                    return response_data["content"]

                # Handle tool calls in the response
                if any(
                    block.get("type") == "tool_use"
                    for block in response_data.get("content", [])
                ):
                    tool_blocks = [
                        block
                        for block in response_data.get("content", [])
                        if block.get("type") == "tool_use"
                    ]
                    tool_calls = []
                    for block in tool_blocks:
                        # Fix: block itself is the tool_use structure, not block["tool_use"]
                        # Anthropic format: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                        tool_calls.append(
                            {
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]) if isinstance(block["input"], dict) else block["input"],
                                },
                            }
                        )

                    if tool_calls:
                        return json.dumps(
                            {"type": "tool_calls", "tool_calls": tool_calls}
                        )

                # Handle thinking in the response
                thinking_content = None
                thinking_models = [
                    "claude-3-7-sonnet-latest",
                    "claude-sonnet-4-20250514",
                    "claude-sonnet-4-0",
                    "claude-opus-4-20250514",
                    "claude-opus-4-0",
                    "claude-sonnet-4-5-20250929",
                    "claude-sonnet-4-5",
                    "claude-opus-4-1-20250805",
                    "claude-opus-4-1",
                    "claude-opus-4-5-20251101",
                    "claude-opus-4-5",
                ]
                if self.valves.ENABLE_THINKING and model_name in thinking_models:
                    thinking_blocks = [
                        block
                        for block in response_data.get("content", [])
                        if block.get("type") == "thinking"
                    ]
                    if thinking_blocks:
                        thinking_content = thinking_blocks[0].get("thinking", "")

                # Get the text response
                text_blocks = [
                    block
                    for block in response_data.get("content", [])
                    if block.get("type") == "text"
                ]
                response_text = text_blocks[0]["text"] if text_blocks else ""

                # If thinking is available, wrap it with <thinking> tags and prepend to the response
                if thinking_content:
                    response_text = (
                        f"<thinking>{thinking_content}</thinking>{response_text}"
                    )

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Request completed successfully",
                                "done": True,
                            },
                        }
                    )

                return response_text

            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                if self.request_id:
                    error_msg += f" (Request ID: {self.request_id})"

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                return {"content": error_msg, "format": "text"}

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

    async def _agentic_tool_loop(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> AsyncIterator[str]:
        """
        Agentic loop: non-streaming call to get tools, execute them, then stream interpretation.
        """
        try:
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Thinking...", "done": False}})

            # Step 1: Non-streaming call to get tool_use blocks
            first_payload = payload.copy()
            first_payload["stream"] = False

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                async with session.post(url, headers=headers, json=first_payload, timeout=timeout) as response:
                    if response.status != 200:
                        error = await response.text()
                        if __event_emitter__:
                            await __event_emitter__({"type": "status", "data": {"description": "Error", "done": True}})
                        yield f"Error: {error}"
                        return

                    response_data = await response.json()

            # Step 2: Check for tool_use blocks
            content_blocks = response_data.get("content", [])
            tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
            text_blocks = [b for b in content_blocks if b.get("type") == "text"]

            # Yield any text from the first response
            for block in text_blocks:
                yield block.get("text", "")

            if not tool_use_blocks:
                # No tools called, we're done
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": "Done", "done": True}})
                return

            # Step 3: Execute tools
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Executing tools...", "done": False}})

            yield "\n\n"
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.get("name", "")
                tool_id = tool_block.get("id", "")
                tool_input = tool_block.get("input", {})

                yield f"**Tool:** `{tool_name}`\n"
                result = await self.execute_mcp_tool(tool_name, tool_input)
                yield f"```\n{result[:1000]}{'...' if len(result) > 1000 else ''}\n```\n\n"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result
                })

            # Step 4: Build follow-up messages
            messages = payload["messages"].copy()

            # Add assistant message with tool_use
            assistant_content = []
            for block in content_blocks:
                if block.get("type") == "text":
                    assistant_content.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {})
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Add user message with tool results
            messages.append({"role": "user", "content": tool_results})

            # Step 5: Stream the interpretation
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Interpreting...", "done": False}})

            followup_payload = {
                "model": payload["model"],
                "messages": messages,
                "max_tokens": payload.get("max_tokens", 8192),
                "stream": True
            }
            if "system" in payload:
                followup_payload["system"] = payload["system"]

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=followup_payload, timeout=timeout) as response:
                    if response.status != 200:
                        error = await response.text()
                        yield f"\n\nInterpretation error: {error}"
                        return

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                text = line[6:].decode("utf-8").strip()
                                if text == "[DONE]":
                                    break
                                data = json.loads(text)
                                if data.get("type") == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        yield delta.get("text", "")
                            except:
                                continue

            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Done", "done": True}})

        except Exception as e:
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Error", "done": True}})
            yield f"\n\nError in agentic loop: {str(e)}"

    async def _stream_with_ui(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> AsyncIterator[str]:
        """
        Stream responses from the Anthropic API with UI event updates.
        Yields text chunks including extended thinking tokens.
        """
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                async with session.post(
                    url, headers=headers, json=payload, timeout=timeout
                ) as response:
                    self.request_id = response.headers.get("x-request-id")
                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"Error: HTTP {response.status}: {error_text}"
                        if self.request_id:
                            error_msg += f" (Request ID: {self.request_id})"
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": error_msg, "done": True},
                                }
                            )
                        yield error_msg
                        return

                    is_thinking = False
                    is_tool_use = False
                    current_tool = {"id": "", "name": "", "input_json": ""}
                    tool_calls = []

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                line_text = line[6:].decode("utf-8").strip()
                                if line_text == "[DONE]":
                                    if is_thinking:
                                        yield "</thinking>"
                                    break

                                data = json.loads(line_text)

                                # Start of a new content block
                                if data["type"] == "content_block_start":
                                    block_type = data.get("content_block", {}).get(
                                        "type"
                                    )
                                    if block_type == "thinking":
                                        is_thinking = True
                                        is_tool_use = False
                                        if self.valves.ENABLE_THINKING:
                                            yield "<thinking>"
                                    elif block_type == "text":
                                        is_thinking = False
                                        is_tool_use = False
                                    elif block_type == "tool_use":
                                        is_thinking = False
                                        is_tool_use = True
                                        content_block = data.get("content_block", {})
                                        current_tool = {
                                            "id": content_block.get("id", ""),
                                            "name": content_block.get("name", ""),
                                            "input_json": ""
                                        }

                                # Handling deltas inside a block
                                elif data["type"] == "content_block_delta":
                                    delta_type = data.get("delta", {}).get("type")
                                    if (
                                        is_thinking
                                        and delta_type == "thinking_delta"
                                        and self.valves.ENABLE_THINKING
                                    ):
                                        yield data["delta"].get("thinking", "")
                                    elif delta_type == "text_delta":
                                        yield data["delta"].get("text", "")
                                    elif delta_type == "input_json_delta" and is_tool_use:
                                        current_tool["input_json"] += data["delta"].get("partial_json", "")

                                # End of the current content block
                                elif data["type"] == "content_block_stop":
                                    if is_thinking:
                                        yield "</thinking>"
                                        is_thinking = False
                                    if is_tool_use:
                                        # Parse the accumulated JSON and add to tool_calls
                                        try:
                                            input_data = json.loads(current_tool["input_json"]) if current_tool["input_json"] else {}
                                        except json.JSONDecodeError:
                                            input_data = {}
                                        tool_calls.append({
                                            "id": current_tool["id"],
                                            "type": "function",
                                            "function": {
                                                "name": current_tool["name"],
                                                "arguments": json.dumps(input_data)
                                            }
                                        })
                                        is_tool_use = False
                                        current_tool = {"id": "", "name": "", "input_json": ""}

                                elif data["type"] == "message_stop":
                                    # If we collected tool calls, execute them and get Claude's interpretation
                                    if tool_calls and self.valves.EXECUTE_TOOLS_IN_PIPE:
                                        # Execute all tools and collect results
                                        tool_results = []
                                        for tc in tool_calls:
                                            func = tc.get("function", {})
                                            tool_name = func.get("name", "")
                                            tool_id = tc.get("id", "")
                                            try:
                                                args = json.loads(func.get("arguments", "{}"))
                                            except:
                                                args = {}

                                            result = await self.execute_mcp_tool(tool_name, args)
                                            tool_results.append({
                                                "type": "tool_result",
                                                "tool_use_id": tool_id,
                                                "content": result
                                            })

                                        # Build follow-up messages with tool results
                                        followup_messages = payload["messages"].copy()
                                        # Add assistant message with tool_use blocks
                                        assistant_tool_use = []
                                        for tc in tool_calls:
                                            func = tc.get("function", {})
                                            try:
                                                input_data = json.loads(func.get("arguments", "{}"))
                                            except:
                                                input_data = {}
                                            assistant_tool_use.append({
                                                "type": "tool_use",
                                                "id": tc.get("id", ""),
                                                "name": func.get("name", ""),
                                                "input": input_data
                                            })
                                        followup_messages.append({
                                            "role": "assistant",
                                            "content": assistant_tool_use
                                        })
                                        # Add user message with tool results and prompt to interpret
                                        tool_results.append({
                                            "type": "text",
                                            "text": "Please interpret and summarize the tool result above."
                                        })
                                        followup_messages.append({
                                            "role": "user",
                                            "content": tool_results
                                        })

                                        # Make follow-up API call
                                        followup_payload = payload.copy()
                                        followup_payload["messages"] = followup_messages
                                        followup_payload["stream"] = True  # Ensure streaming
                                        # Remove tools to avoid infinite loop
                                        followup_payload.pop("tools", None)
                                        followup_payload.pop("tool_choice", None)
                                        # Disable extended thinking for follow-up (avoids thinking block requirement)
                                        followup_payload.pop("thinking", None)

                                        yield "\n\n"

                                        # Stream the follow-up response
                                        try:
                                            with open('/tmp/aurora_followup.log', 'a') as f:
                                                f.write(f"\n--- Follow-up at {datetime.now()} ---\n")
                                                f.write(f"Messages count: {len(followup_messages)}\n")
                                                f.write(f"Last message role: {followup_messages[-1].get('role')}\n")
                                                f.write(f"Last message content: {str(followup_messages[-1].get('content'))[:500]}\n")

                                            async with aiohttp.ClientSession() as followup_session:
                                                async with followup_session.post(
                                                    url, headers=headers, json=followup_payload, timeout=timeout
                                                ) as followup_response:
                                                    with open('/tmp/aurora_followup.log', 'a') as f:
                                                        f.write(f"Follow-up status: {followup_response.status}\n")

                                                    if followup_response.status == 200:
                                                        line_count = 0
                                                        async for followup_line in followup_response.content:
                                                            line_count += 1
                                                            if line_count <= 5:
                                                                with open('/tmp/aurora_followup.log', 'a') as f:
                                                                    f.write(f"Line {line_count}: {followup_line[:100]}\n")
                                                            if followup_line and followup_line.startswith(b"data: "):
                                                                try:
                                                                    followup_text = followup_line[6:].decode("utf-8").strip()
                                                                    if followup_text == "[DONE]":
                                                                        break
                                                                    followup_data = json.loads(followup_text)
                                                                    with open('/tmp/aurora_followup.log', 'a') as f:
                                                                        f.write(f"Event type: {followup_data.get('type')}\n")
                                                                    if followup_data["type"] == "content_block_delta":
                                                                        delta_type = followup_data.get("delta", {}).get("type")
                                                                        if delta_type == "text_delta":
                                                                            text = followup_data["delta"].get("text", "")
                                                                            with open('/tmp/aurora_followup.log', 'a') as f:
                                                                                f.write(f"Yielding: {text[:50]}\n")
                                                                            yield text
                                                                except Exception as parse_err:
                                                                    with open('/tmp/aurora_followup.log', 'a') as f:
                                                                        f.write(f"Parse error: {parse_err}\n")
                                                                    continue
                                                        with open('/tmp/aurora_followup.log', 'a') as f:
                                                            f.write(f"Total lines: {line_count}\n")
                                                    else:
                                                        error_body = await followup_response.text()
                                                        with open('/tmp/aurora_followup.log', 'a') as f:
                                                            f.write(f"Follow-up error: {error_body}\n")
                                                        yield f"\n\n[Follow-up failed: HTTP {followup_response.status}]"
                                        except Exception as followup_err:
                                            with open('/tmp/aurora_followup.log', 'a') as f:
                                                f.write(f"Follow-up exception: {followup_err}\n")
                                            yield f"\n\n[Follow-up error: {followup_err}]"
                                    elif tool_calls:
                                        # Fallback: just output the tool calls as JSON
                                        yield json.dumps({"type": "tool_calls", "tool_calls": tool_calls})
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "Request completed",
                                                    "done": True,
                                                },
                                            }
                                        )
                                    break
                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
                                continue
        except asyncio.TimeoutError:
            error_msg = "Request timed out"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            yield error_msg
        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            yield error_msg

    def _process_messages(self, messages: List[dict]) -> List[dict]:
        """
        Process messages for the Anthropic API format.

        Args:
            messages: List of message objects

        Returns:
            Processed messages in Anthropic API format
        """
        processed_messages = []
        for message in messages:
            processed_content = []
            for content in self.process_content(message["content"]):
                if (
                    message.get("role") == "assistant"
                    and content.get("type") == "tool_calls"
                ):
                    content["cache_control"] = {"type": "ephemeral"}
                elif (
                    message.get("role") == "user"
                    and content.get("type") == "tool_results"
                ):
                    content["cache_control"] = {"type": "ephemeral"}
                processed_content.append(content)
            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )
        return processed_messages

    async def _send_request(
        self, url: str, headers: dict, payload: dict
    ) -> tuple[dict, Optional[dict]]:
        """
        Send a request to the Anthropic API with retry logic.

        Args:
            url: The API endpoint URL
            headers: Request headers
            payload: Request payload

        Returns:
            Tuple of (response_data, cache_metrics)
        """
        retry_count = 0
        base_delay = 1  # Start with 1 second delay
        max_retries = 3

        while retry_count < max_retries:
            try:
                async with aiohttp.ClientSession() as session:
                    timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                    async with session.post(
                        url, headers=headers, json=payload, timeout=timeout
                    ) as response:
                        self.request_id = response.headers.get("x-request-id")

                        if response.status == 429:
                            retry_after = int(
                                response.headers.get(
                                    "retry-after", base_delay * (2**retry_count)
                                )
                            )
                            logging.warning(
                                f"Rate limit hit. Retrying in {retry_after} seconds. Retry count: {retry_count + 1}"
                            )
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue

                        response_text = await response.text()

                        if response.status != 200:
                            error_msg = f"Error: HTTP {response.status}"
                            try:
                                error_data = json.loads(response_text).get("error", {})
                                error_msg += (
                                    f": {error_data.get('message', response_text)}"
                                )
                            except:
                                error_msg += f": {response_text}"

                            if self.request_id:
                                error_msg += f" (Request ID: {self.request_id})"

                            return {"content": error_msg, "format": "text"}, None

                        result = json.loads(response_text)
                        usage = result.get("usage", {})
                        cache_metrics = {
                            "cache_creation_input_tokens": usage.get(
                                "cache_creation_input_tokens", 0
                            ),
                            "cache_read_input_tokens": usage.get(
                                "cache_read_input_tokens", 0
                            ),
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0),
                        }
                        return result, cache_metrics

            except aiohttp.ClientError as e:
                logging.error(f"Request failed: {str(e)}")
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(base_delay * (2**retry_count))
                    continue
                raise

        logging.error("Max retries exceeded for rate limit.")
        return {"content": "Max retries exceeded", "format": "text"}, None


