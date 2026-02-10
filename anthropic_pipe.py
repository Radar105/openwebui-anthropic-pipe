"""
title: Anthropic API Integration - Claude 4, 4.5, Opus 4.5, Opus 4.6 + Extended Thinking
author: Balaxxe (original), updated by Radar105
version: 6.0
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, aiohttp
environment_variables:
    - ANTHROPIC_API_KEY (required)

Supports:
- All Claude 3, 3.5, 3.7, 4, 4.5, Opus 4.1, Opus 4.5, and Opus 4.6 models
- Streaming responses
- Image processing
- Prompt caching (server-side)
- Tool calling (OpenAI-format delta.tool_calls for Open WebUI v0.7 native handling)
- PDF processing
- Cache Control
- Extended thinking (Claude 3.7+, 4+)

Architecture (Tool Calling):
- Pipe translates between Anthropic and OpenAI formats
- Open WebUI v0.7 handles the full tool execution loop natively
- Pipe yields OpenAI-format delta.tool_calls chunks during streaming
- Middleware executes tools via configured tool_server connections (MCPO)
- Middleware re-calls pipe with tool results as role=tool messages

Updates:
v6.0 - Removed in-pipe tool execution, aligned with Open WebUI v0.7 native tool handling
v5.1 - Added Claude Opus 4.6 (Feb 5, 2026)
v5.0 - Agentic loop for tool execution, Claude Opus 4.5, MCP tool integration
v4.0 - Added Claude 4, Claude 4.5 Sonnet, Opus 4.1
v3.0 - Added Claude 3.7, thinking valves, CoT streaming
"""

import os
import json
import time
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
        "claude-opus-4-6",
    ]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024
    PDF_BETA_HEADER = "pdfs-2024-09-25"
    OUTPUT_128K_BETA = "output-128k-2025-02-19"
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
        "claude-3-7-sonnet-latest": 16384,
        "claude-sonnet-4-20250514": 16384,
        "claude-sonnet-4-0": 16384,
        "claude-opus-4-20250514": 16384,
        "claude-opus-4-0": 16384,
        "claude-sonnet-4-5-20250929": 32768,
        "claude-sonnet-4-5": 32768,
        "claude-opus-4-1-20250805": 32768,
        "claude-opus-4-1": 32768,
        "claude-opus-4-5-20251101": 32768,
        "claude-opus-4-5": 32768,
        "claude-opus-4-6": 32768,
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
        "claude-3-7-sonnet-latest": 200000,
        "claude-sonnet-4-20250514": 200000,
        "claude-sonnet-4-0": 200000,
        "claude-opus-4-20250514": 200000,
        "claude-opus-4-0": 200000,
        "claude-sonnet-4-5-20250929": 200000,
        "claude-sonnet-4-5": 200000,
        "claude-opus-4-1-20250805": 200000,
        "claude-opus-4-1": 200000,
        "claude-opus-4-5-20251101": 200000,
        "claude-opus-4-5": 200000,
        "claude-opus-4-6": 200000,
    }
    BETA_HEADER = "prompt-caching-2024-07-31"
    REQUEST_TIMEOUT = 120
    THINKING_BUDGET_TOKENS = 16000

    THINKING_MODELS = [
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
        "claude-opus-4-6",
    ]

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = "Your API Key Here"
        ENABLE_THINKING: bool = False
        MAX_OUTPUT_TOKENS: bool = True
        ENABLE_TOOL_CHOICE: bool = True
        ENABLE_SYSTEM_PROMPT: bool = True
        THINKING_BUDGET_TOKENS: int = Field(
            default=16000, ge=0, le=16000
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None

    def get_anthropic_models(self) -> List[dict]:
        return [
            {
                "id": f"anthropic/{name}",
                "name": name,
                "context_length": self.MODEL_CONTEXT_LENGTH.get(name, 200000),
                "supports_vision": name != "claude-3-5-haiku-20241022",
                "supports_thinking": name in self.THINKING_MODELS,
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
                "claude-opus-4-6",
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
        return processed_content

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            if media_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported media type: {media_type}")

            image_size = len(base64_data) * 3 / 4
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

            pdf_size = len(base64_data) * 3 / 4
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
        if not self.valves.ANTHROPIC_API_KEY:
            error_msg = "Error: ANTHROPIC_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}

        try:
            system_message, messages = pop_system_message(body["messages"])

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": "Processing request...", "done": False}}
                )

            model_name = body["model"].split("/")[-1]
            if model_name not in self.MODEL_MAX_TOKENS:
                logging.warning(f"Unknown model: {model_name}, using default token limit")

            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_name, 4096)

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

            if self.valves.ENABLE_THINKING and model_name in self.THINKING_MODELS:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": self.valves.THINKING_BUDGET_TOKENS,
                }

            payload = {k: v for k, v in payload.items() if v is not None}

            if system_message and self.valves.ENABLE_SYSTEM_PROMPT:
                payload["system"] = str(system_message)

            # Convert OpenAI tool format to Anthropic format
            if "tools" in body and self.valves.ENABLE_TOOL_CHOICE:
                anthropic_tools = []
                for tool in body["tools"]:
                    if isinstance(tool, dict) and tool.get("type") == "function":
                        func = tool.get("function", {})
                        anthropic_tools.append({
                            "name": func.get("name"),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                        })
                    elif isinstance(tool, dict) and "name" in tool:
                        anthropic_tools.append({
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "input_schema": tool.get("parameters", {"type": "object", "properties": {}})
                        })
                if anthropic_tools:
                    payload["tools"] = anthropic_tools
                tool_choice = body.get("tool_choice")
                if tool_choice:
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            payload["tool_choice"] = {"type": "auto"}
                        elif tool_choice in ("required", "any"):
                            payload["tool_choice"] = {"type": "any"}
                    elif isinstance(tool_choice, dict):
                        payload["tool_choice"] = tool_choice
                else:
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

            if any(
                isinstance(msg["content"], list)
                and any(item.get("type") == "pdf_url" for item in msg["content"])
                for msg in body.get("messages", [])
            ):
                beta_headers.append(self.PDF_BETA_HEADER)

            if any(
                isinstance(msg["content"], list)
                and any(item.get("cache_control") for item in msg["content"])
                for msg in body.get("messages", [])
            ):
                beta_headers.append(self.BETA_HEADER)

            if model_name in self.THINKING_MODELS and self.valves.MAX_OUTPUT_TOKENS:
                beta_headers.append(self.OUTPUT_128K_BETA)

            if beta_headers:
                headers["anthropic-beta"] = ",".join(beta_headers)

            try:
                if payload.get("stream", False):
                    return self._stream_with_ui(
                        self.MODEL_URL, headers, payload, body, __event_emitter__
                    )

                # Non-streaming path
                response_data, cache_metrics = await self._send_request(
                    self.MODEL_URL, headers, payload
                )

                if (
                    isinstance(response_data, dict)
                    and "content" in response_data
                    and response_data.get("format") == "text"
                ):
                    if __event_emitter__:
                        await __event_emitter__(
                            {"type": "status", "data": {"description": response_data["content"], "done": True}}
                        )
                    return response_data["content"]

                result = self._format_openai_response(response_data, model_name)

                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": "Request completed successfully", "done": True}}
                    )

                return result

            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                if self.request_id:
                    error_msg += f" (Request ID: {self.request_id})"
                if __event_emitter__:
                    await __event_emitter__(
                        {"type": "status", "data": {"description": error_msg, "done": True}}
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

    def _format_openai_response(self, response_data: dict, model_name: str) -> Union[str, dict]:
        """Format an Anthropic non-streaming response as OpenAI format."""
        content_blocks = response_data.get("content", [])
        text_blocks = [b for b in content_blocks if b.get("type") == "text"]
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
        thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]

        text_content = ""
        if thinking_blocks and self.valves.ENABLE_THINKING:
            text_content += f"<thinking>{thinking_blocks[0].get('thinking', '')}</thinking>"
        if text_blocks:
            text_content += text_blocks[0].get("text", "")

        # If tool_use blocks present, return OpenAI-format dict with tool_calls
        if tool_use_blocks:
            msg_id = response_data.get("id", "unknown")
            message = {"role": "assistant", "content": text_content}
            tool_calls = []
            for i, block in enumerate(tool_use_blocks):
                tool_calls.append({
                    "index": i,
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]) if isinstance(block["input"], dict) else block["input"]
                    }
                })
            message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{msg_id}",
                "object": "chat.completion",
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls"
                }]
            }

        # No tools — return plain text
        return text_content

    async def _stream_with_ui(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> AsyncIterator:
        """
        Stream Anthropic SSE events, translating to OpenAI format.
        Yields strings for text content and dicts for tool_calls chunks.
        Open WebUI's process_line handles both: strings get wrapped in
        openai_chat_chunk_message_template, dicts pass through as raw SSE.
        """
        model_name = body["model"].split("/")[-1]
        msg_id = None
        tool_call_index = -1

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
                                {"type": "status", "data": {"description": error_msg, "done": True}}
                            )
                        yield error_msg
                        return

                    is_thinking = False

                    async for line in response.content:
                        if not line or not line.startswith(b"data: "):
                            continue
                        try:
                            line_text = line[6:].decode("utf-8").strip()
                            if line_text == "[DONE]":
                                break

                            data = json.loads(line_text)
                            event_type = data.get("type", "")

                            if event_type == "message_start":
                                msg_id = data.get("message", {}).get("id", "unknown")

                            elif event_type == "content_block_start":
                                block = data.get("content_block", {})
                                block_type = block.get("type")

                                if block_type == "thinking":
                                    is_thinking = True
                                    if self.valves.ENABLE_THINKING:
                                        yield "<thinking>"

                                elif block_type == "text":
                                    is_thinking = False

                                elif block_type == "tool_use":
                                    is_thinking = False
                                    tool_call_index += 1
                                    # Yield initial tool_call chunk with id and name
                                    yield {
                                        "id": f"chatcmpl-{msg_id}",
                                        "object": "chat.completion.chunk",
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [{
                                                    "index": tool_call_index,
                                                    "id": block.get("id", ""),
                                                    "type": "function",
                                                    "function": {
                                                        "name": block.get("name", ""),
                                                        "arguments": ""
                                                    }
                                                }]
                                            },
                                            "finish_reason": None
                                        }]
                                    }

                            elif event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                delta_type = delta.get("type")

                                if delta_type == "thinking_delta" and is_thinking and self.valves.ENABLE_THINKING:
                                    yield delta.get("thinking", "")

                                elif delta_type == "text_delta":
                                    yield delta.get("text", "")

                                elif delta_type == "input_json_delta":
                                    partial = delta.get("partial_json", "")
                                    if partial:
                                        yield {
                                            "id": f"chatcmpl-{msg_id}",
                                            "object": "chat.completion.chunk",
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "tool_calls": [{
                                                        "index": tool_call_index,
                                                        "function": {
                                                            "arguments": partial
                                                        }
                                                    }]
                                                },
                                                "finish_reason": None
                                            }]
                                        }

                            elif event_type == "content_block_stop":
                                if is_thinking and self.valves.ENABLE_THINKING:
                                    yield "</thinking>"
                                    is_thinking = False

                            elif event_type == "message_stop":
                                if __event_emitter__:
                                    await __event_emitter__(
                                        {"type": "status", "data": {"description": "Done", "done": True}}
                                    )

                        except json.JSONDecodeError:
                            continue

        except asyncio.TimeoutError:
            error_msg = "Request timed out"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg
        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield error_msg

    def _process_messages(self, messages: List[dict]) -> List[dict]:
        """
        Process messages to Anthropic API format.
        Handles standard messages, OpenAI-format assistant+tool_calls,
        and role=tool result messages from Open WebUI's tool loop.
        """
        processed = []
        i = 0
        while i < len(messages):
            msg = messages[i]

            # Assistant message with tool_calls (from middleware tool loop)
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                content_blocks = []
                text = msg.get("content", "")
                if text:
                    content_blocks.append({"type": "text", "text": text})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    try:
                        input_data = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        input_data = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": input_data
                    })
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": ""})
                processed.append({"role": "assistant", "content": content_blocks})
                i += 1

            # Tool result messages (from middleware tool loop)
            elif msg.get("role") == "tool":
                tool_results = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_msg.get("tool_call_id", ""),
                        "content": tool_msg.get("content", "")
                    })
                    i += 1
                processed.append({"role": "user", "content": tool_results})

            # Normal message
            else:
                processed_content = self.process_content(msg["content"])
                processed.append({"role": msg["role"], "content": processed_content})
                i += 1

        return processed

    async def _send_request(
        self, url: str, headers: dict, payload: dict
    ) -> tuple[dict, Optional[dict]]:
        """Send a request to the Anthropic API with retry logic."""
        retry_count = 0
        base_delay = 1
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
