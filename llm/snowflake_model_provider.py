"""Snowflake Cortex model provider for the Strands agent framework.

Exposes a Strands-compatible ``Model`` adapter that delegates inference to
Snowflake Cortex, with support for streaming events, tool-call parsing, and
multimodal content upload via Snowflake internal stages.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncIterable, Optional, TypedDict

from typing_extensions import Unpack, override

from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec

try:
    from .llm_snowflake import SnowflakeCortexClient
except ImportError:
    from llm.llm_snowflake import SnowflakeCortexClient

logger = logging.getLogger(__name__)


class SnowflakeStrandsModel(Model):
    """Strands-compatible adapter for Snowflake Cortex LLM inference.

    Wraps a ``SnowflakeCortexClient`` to conform to the Strands ``Model``
    interface, providing streaming responses, tool-call parsing, and
    multimodal content handling via Snowflake internal stages.
    """

    class ModelConfig(TypedDict):
        """Typed configuration dictionary for model parameters."""

        model: Optional[str]
        temperature: float

    def __init__(
        self,
        llm_client: Optional[SnowflakeCortexClient] = None,
        model_id: Optional[str] = None,
        **model_config: Unpack["SnowflakeStrandsModel.ModelConfig"],
    ) -> None:
        """Initialize the Snowflake Strands model adapter.

        Args:
            llm_client: Pre-configured Snowflake Cortex client. A default
                client is created if not provided.
            model_id: Shorthand for setting the model name in config.
            **model_config: Additional model configuration overrides.
        """
        self.llm_client = llm_client or SnowflakeCortexClient()

        if model_id and "model" not in model_config:
            model_config["model"] = model_id

        default_config = {"model": "openai-gpt-5.2", "temperature": 0.0}
        default_config.update(model_config)
        self.config = self.ModelConfig(**default_config)

    def close(self) -> None:
        """Release the underlying Snowflake client connection."""
        self.llm_client.close()

    def update_config(
        self, **model_config: Unpack["SnowflakeStrandsModel.ModelConfig"]
    ) -> None:
        """Merge new values into the current model configuration.

        Args:
            **model_config: Key-value pairs to update.
        """
        self.config.update(model_config)

    def get_config(self) -> "SnowflakeStrandsModel.ModelConfig":
        """Return the current model configuration dictionary."""
        return self.config

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream responses from Snowflake Cortex with tool-call support.

        Extracts multimodal content, builds the prompt with tool definitions,
        invokes the LLM synchronously in a background thread, and yields
        Strands-compatible stream events.

        Args:
            messages: Conversation history in Strands message format.
            tool_specs: Optional tool definitions to inject into the prompt.
            system_prompt: Optional caller-provided system prompt.
            **kwargs: Additional keyword arguments (forwarded).

        Yields:
            StreamEvent dicts consumed by the Strands runtime.
        """
        image_urls = await asyncio.to_thread(self._extract_and_upload_images, messages)

        tool_system, user_prompt = self._format_messages_to_prompt(messages, tool_specs)

        full_system = "\n\n".join(p for p in [system_prompt, tool_system] if p)

        response_text = await asyncio.to_thread(
            self.llm_client.call,
            model=self.config.get("model", "openai-gpt-5.2"),
            system_prompt=full_system,
            user_prompt=user_prompt,
            temperature=self.config.get("temperature", 0.0),
            image_urls=image_urls or None,
        )

        if not response_text:
            response_text = ""

        async for event in self._emit_events(response_text, tool_specs):
            yield event

    # ------------------------------------------------------------------ #
    #  Response Parsing & Event Emission
    # ------------------------------------------------------------------ #

    async def _emit_events(
        self, response_text: str, tool_specs: Optional[list[ToolSpec]]
    ) -> AsyncIterable[StreamEvent]:
        """Parse an LLM response and yield Strands StreamEvent dicts.

        Detects JSON tool-call blocks in the response text.  When found, each
        call is emitted as a ``toolUse`` content block; otherwise the full
        text is emitted as a plain content block.

        Args:
            response_text: Raw text returned by the LLM.
            tool_specs: Tool definitions used for name validation.

        Yields:
            StreamEvent dicts conforming to the Strands streaming protocol.
        """
        yield {"messageStart": {"role": "assistant"}}

        tool_calls = self._extract_tool_calls(response_text)

        if tool_calls:
            leading_text = self._extract_leading_text(response_text)
            if leading_text:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": leading_text}}}
                yield {"contentBlockStop": {}}

            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                tool_args = tc.get("parameters", {})
                tool_use_id = f"tooluse_{uuid.uuid4().hex[:12]}"

                if tool_specs and not any(
                    s.get("name") == tool_name for s in tool_specs
                ):
                    logger.warning(f"LLM called unknown tool '{tool_name}', skipping")
                    continue

                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "name": tool_name,
                                "toolUseId": tool_use_id,
                            }
                        }
                    }
                }
                yield {
                    "contentBlockDelta": {
                        "delta": {"toolUse": {"input": json.dumps(tool_args)}}
                    }
                }
                yield {"contentBlockStop": {}}

            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            text = response_text.strip()
            if text:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": text}}}
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    def _extract_tool_calls(self, text: str) -> list[dict]:
        """Extract JSON tool-call objects from raw LLM output.

        Strips markdown code fences, then scans for the first valid JSON
        object or array containing ``{"tool": ..., "parameters": ...}``
        entries.

        Args:
            text: Raw response string from the LLM.

        Returns:
            List of tool-call dicts, empty if none found.
        """
        clean = text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        calls = []
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            idx = clean.find(start_char)
            if idx == -1:
                continue
            depth = 0
            end_idx = -1
            for i in range(idx, len(clean)):
                if clean[i] == start_char:
                    depth += 1
                elif clean[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx == -1:
                continue
            try:
                parsed = json.loads(clean[idx : end_idx + 1])
                if isinstance(parsed, list):
                    calls.extend(
                        obj for obj in parsed if isinstance(obj, dict) and "tool" in obj
                    )
                elif isinstance(parsed, dict) and "tool" in parsed:
                    calls.append(parsed)
                if calls:
                    return calls
            except json.JSONDecodeError:
                continue
        return []

    @staticmethod
    def _extract_leading_text(response_text: str) -> str:
        """Return text preceding the first JSON object or array in the response.

        Args:
            response_text: Full LLM response string.

        Returns:
            Leading plain text, or an empty string if none exists.
        """
        for ch in ("{", "["):
            idx = response_text.find(ch)
            if idx > 0:
                leading = response_text[:idx].strip()
                if leading:
                    return leading
        return ""

    # ------------------------------------------------------------------ #
    #  Multimodal Content Extraction & Stage Upload
    # ------------------------------------------------------------------ #

    def _extract_and_upload_images(self, messages: Messages) -> list[str]:
        """Upload image and document payloads from messages to a Snowflake stage.

        Args:
            messages: Strands conversation messages potentially containing
                binary image or document content blocks.

        Returns:
            List of ``@stage/filename`` references for uploaded assets.
        """
        urls: list[str] = []
        for msg in messages:
            for block in msg.get("content", []):
                if "image" in block:
                    img = block["image"]
                    fmt = img.get("format", "png")
                    data = img["source"]["bytes"]
                    filename = f"agent_img_{uuid.uuid4().hex[:8]}.{fmt}"
                    try:
                        url = self.llm_client.upload_bytes_to_stage(data, filename)
                        urls.append(url)
                        logger.info("Uploaded image to stage: %s", url)
                    except Exception as exc:
                        logger.warning("Failed to upload image to stage: %s", exc)

                elif "document" in block:
                    doc = block["document"]
                    fmt = doc.get("format", "pdf")
                    name = doc.get("name", "document")
                    data = doc["source"]["bytes"]
                    filename = f"{name}_{uuid.uuid4().hex[:6]}.{fmt}"
                    try:
                        url = self.llm_client.upload_bytes_to_stage(data, filename)
                        urls.append(url)
                        logger.info("Uploaded document to stage: %s", url)
                    except Exception as exc:
                        logger.warning("Failed to upload document to stage: %s", exc)
        return urls

    async def structured_output(
        self,
        messages: Messages,
        output_schema: Any,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate structured output by streaming and parsing the first text delta.

        Args:
            messages: Conversation history in Strands message format.
            output_schema: Expected output schema (currently unused).
            tool_specs: Optional tool definitions.
            system_prompt: Optional system prompt.
            **kwargs: Forwarded to the stream method.

        Returns:
            Parsed JSON object, raw text string, or ``None``.
        """
        async for event in self.stream(messages, tool_specs, system_prompt, **kwargs):
            if event.get("contentBlockDelta"):
                text = event["contentBlockDelta"]["delta"].get("text", "")
                try:
                    import json

                    return json.loads(text)
                except:
                    return text
        return None

    def _format_messages_to_prompt(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None
    ) -> tuple[str, str]:
        """Convert Strands messages and tool specs into a flat prompt pair.

        Flattens the structured conversation history (text, tool-use records,
        tool results, images, documents) into a single user prompt string and
        builds a system prompt section describing available tools.

        Args:
            messages: Strands conversation messages.
            tool_specs: Optional tool definitions to include in the system prompt.

        Returns:
            Tuple of ``(tool_system_prompt, user_prompt)``.
        """
        conversation_parts = []

        for msg in messages:
            role = msg["role"].upper()
            content_blocks = msg.get("content", [])

            for block in content_blocks:
                if "text" in block:
                    conversation_parts.append(f"{role}: {block['text']}")

                elif "toolUse" in block:
                    tool_use = block["toolUse"]
                    conversation_parts.append(
                        f"{role}: [Called tool '{tool_use['name']}' with parameters: {json.dumps(tool_use['input'])}]"
                    )

                elif "toolResult" in block:
                    tool_result = block["toolResult"]
                    result_text_parts = []

                    for result_content in tool_result.get("content", []):
                        if "text" in result_content:
                            result_text_parts.append(result_content["text"])
                        elif "json" in result_content:
                            result_text_parts.append(
                                json.dumps(result_content["json"], indent=2)
                            )

                    result_text = "\n".join(result_text_parts)
                    conversation_parts.append(f"{role}: [Tool result]: {result_text}")

                elif "image" in block:
                    conversation_parts.append(
                        f"{role}: [Image attached -- sent via stage reference]"
                    )

                elif "document" in block:
                    doc = block["document"]
                    name = doc.get("name", "document")
                    conversation_parts.append(
                        f"{role}: [Document '{name}' attached -- sent via stage reference]"
                    )

        user_prompt = "\n\n".join(conversation_parts)

        system_parts = []

        if tool_specs:
            system_parts.append("=== AVAILABLE TOOLS ===")
            for tool in tool_specs:
                tool_info = f"\nTool: {tool['name']}\n"
                tool_info += f"Description: {tool.get('description', '')}\n"

                if "input_schema" in tool and "properties" in tool["input_schema"]:
                    props = tool["input_schema"]["properties"]
                    tool_info += "Parameters:\n"
                    for param_name, param_details in props.items():
                        param_type = param_details.get("type", "string")
                        param_desc = param_details.get("description", "")
                        tool_info += f"  - {param_name} ({param_type}): {param_desc}\n"

                    required = tool["input_schema"].get("required", [])
                    if required:
                        tool_info += f"Required: {', '.join(required)}\n"

                system_parts.append(tool_info)

            system_parts.append(
                "\n=== HOW TO USE TOOLS ===\n"
                "When you need information from a tool, respond with ONLY a JSON object:\n"
                '{"tool": "tool_name", "parameters": {"param": "value"}}\n\n'
                "CRITICAL RULES:\n"
                "- Output ONLY raw JSON - no markdown, no backticks, no extra text\n"
                "- Call ONE tool at a time\n"
                "- Use correct parameter names as shown above\n"
                "- After receiving tool results, decide: call another tool OR give final answer\n"
                "- NEVER call the same tool twice with the same parameters\n"
                "- Once you have enough information, respond with your final answer as regular text\n"
                "- Your final answer should synthesize the tool results into a helpful response"
            )

        system_prompt = "\n".join(system_parts)
        return system_prompt, user_prompt
