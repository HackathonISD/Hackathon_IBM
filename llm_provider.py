"""Universal LLM provider router with ReAct tool-calling support.

Routes requests to 14+ LLM backends (Snowflake Cortex, OpenAI, Anthropic,
Gemini, Bedrock, Ollama, etc.) through a unified interface. Text-only
providers gain tool-calling capability via a built-in ReAct loop that
parses structured JSON tool invocations from model output.
"""

import os
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("LITELLM_LOG", "ERROR")
try:
    import litellm as _litellm_init

    _litellm_init.suppress_debug_info = True
    _litellm_init.drop_params = True
    _litellm_init.set_verbose = False
except ImportError:
    pass


# ============ REACT TOOL EXECUTOR ============

REACT_SYSTEM_PREFIX = """\
You are an AI agent with access to tools.

TOOL CALL FORMAT — when you need a tool, respond with ONLY this JSON (nothing else before or after):
{{"tool": "<tool_name>", "args": {{<args as JSON object>}}}}

FINAL ANSWER — when done, respond with plain text (no JSON).

Available tools:
{tool_descriptions}

RULES:
- Call ONE tool per response
- After each tool result you receive, decide your next action
- When all work is complete, write a plain text summary
- Never invent tool names not in the list
"""


def _build_tool_descriptions(tool_specs: list) -> str:
    """Format tool specifications into a human-readable description block."""
    lines = []
    for spec in tool_specs or []:
        name = spec.get("name", "unknown")
        desc = spec.get("description", "")
        if not desc:
            desc = (spec.get("inputSchema") or {}).get("description", "")
        props = (spec.get("inputSchema") or {}).get("json", {}).get("properties", {})
        if not props:
            props = (spec.get("parameters") or {}).get("properties", {})
        params = ", ".join(props.keys()) if props else "no params"
        lines.append(f"- {name}({params}): {desc}")
    return "\n".join(lines) or "No tools available"


def _extract_tool_call(text: str) -> Optional[dict]:
    """Extract a tool-call JSON object from LLM response text.

    Tries multiple strategies: markdown fences, direct parse, positional
    scanning with JSONDecoder, and sanitized-string fallback.

    Returns:
        Parsed dict with 'tool' key if found, None otherwise.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1).strip())
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "tool" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    sanitized = _sanitize_json_strings(text)
    if sanitized != text:
        for i, c in enumerate(sanitized):
            if c != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(sanitized, i)
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _sanitize_json_strings(text: str) -> str:
    """Escape unescaped control characters inside JSON string values."""
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
        out.append(ch)
    return "".join(out)


def _execute_tool(tool_name: str, args: dict, tool_map: dict) -> str:
    """Invoke a tool function by name and return its string result."""
    fn = tool_map.get(tool_name)
    if fn is None:
        return (
            f"ERROR: tool '{tool_name}' not found. Available: {list(tool_map.keys())}"
        )
    try:
        result = fn(**args)
        return str(result) if result is not None else "Done"
    except Exception as e:
        return f"ERROR in {tool_name}: {e}"


def run_react_loop(completer, messages, tool_specs, tool_map, max_steps=15):
    """Execute a ReAct reasoning-action loop for text-only LLMs.

    Args:
        completer: Callable that takes a message list and returns a string.
        messages: Initial conversation history.
        tool_specs: Tool specification dicts describing available tools.
        tool_map: Mapping of tool names to callable implementations.
        max_steps: Maximum number of tool-call iterations.

    Returns:
        Final plain-text response from the LLM.
    """
    desc = _build_tool_descriptions(tool_specs)
    react_sys = REACT_SYSTEM_PREFIX.format(tool_descriptions=desc)

    history = list(messages)
    has_sys = any(m.get("role") == "system" for m in history)
    if has_sys:
        for i, m in enumerate(history):
            if m.get("role") == "system":
                history[i] = {
                    "role": "system",
                    "content": react_sys + "\n\n" + m["content"],
                }
                break
    else:
        history.insert(0, {"role": "system", "content": react_sys})

    for _ in range(max_steps):
        response = completer(history)
        tool_call = _extract_tool_call(response)
        if tool_call is None:
            return response
        tool_name = tool_call.get("tool", "")
        tool_args = tool_call.get("args", {})
        result = _execute_tool(tool_name, tool_args, tool_map)
        history.append({"role": "assistant", "content": response})
        history.append(
            {
                "role": "user",
                "content": f"Tool '{tool_name}' returned:\n{result}\n\nContinue.",
            }
        )
    return "Agent reached max steps. Partial work saved."


# ============ BASE PROVIDER ============


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider implementations."""

    def __init__(
        self, model: str, temperature: float = 0.0, max_tokens: Optional[int] = None
    ):
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def complete(
        self, user_prompt: str, system_prompt: Optional[str] = None
    ) -> str: ...

    def complete_messages(self, messages: list) -> str:
        """Flatten a message list into system/user parts and call complete()."""
        system_parts, user_parts = [], []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if role == "system":
                system_parts.append(content)
            else:
                prefix = "Assistant: " if role == "assistant" else "User: "
                user_parts.append(prefix + content)
        return self.complete(
            user_prompt="\n\n".join(user_parts),
            system_prompt="\n\n".join(system_parts) or None,
        )

    def to_strands_model(self):
        """Return a Strands-compatible model, falling back to ReAct wrapper."""
        try:
            return self._make_strands_model()
        except Exception:
            return StrandsModelWrapper(self)

    def _make_strands_model(self):
        """Create a native Strands model adapter for this provider."""
        raise NotImplementedError


# ============ STRANDS REACT WRAPPER ============


class StrandsModelWrapper:
    """Strands-compatible model wrapper with ReAct tool-calling for text-only LLMs.

    Parses structured JSON tool invocations from plain-text LLM output and
    emits proper toolUse stream events for the Strands agent loop.
    """

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider
        self.config = {
            "model_id": provider.model,
            "temperature": provider.temperature,
            "max_tokens": provider.max_tokens,
        }

    def update_config(self, **model_config: Any) -> None:
        """Merge additional configuration into the model config."""
        self.config.update(model_config)

    def get_config(self) -> dict:
        """Return the current model configuration."""
        return self.config

    @staticmethod
    def _flatten_messages(messages, system_prompt=None, tool_specs=None):
        """Convert Strands-format messages to simple role/content dicts."""
        plain = []

        tool_section = ""
        if tool_specs:
            tool_section = _build_tool_descriptions(tool_specs)

        sys_prefix = (
            REACT_SYSTEM_PREFIX.format(tool_descriptions=tool_section)
            if tool_specs
            else ""
        )

        if system_prompt:
            sys_prefix = (
                sys_prefix + "\n\n" + system_prompt if sys_prefix else system_prompt
            )

        if sys_prefix:
            plain.append({"role": "system", "content": sys_prefix})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        if "text" in block:
                            parts.append(block["text"])
                        elif "toolUse" in block:
                            tu = block["toolUse"]
                            parts.append(
                                f'[Called tool {tu.get("name", "?")}({json.dumps(tu.get("input", {}))})]\n'
                            )
                        elif "toolResult" in block:
                            tr = block["toolResult"]
                            result_parts = []
                            for c in tr.get("content", []):
                                if isinstance(c, dict) and "text" in c:
                                    result_parts.append(c["text"])
                                elif isinstance(c, dict) and "json" in c:
                                    result_parts.append(json.dumps(c["json"]))
                                else:
                                    result_parts.append(str(c))
                            status = tr.get("status", "success")
                            parts.append(
                                f'[Tool result ({status}): {" ".join(result_parts)}]\n'
                            )
                content = "\n".join(parts)
            if not content or not content.strip():
                continue
            if role == "system" and plain and plain[0]["role"] == "system":
                plain[0]["content"] += "\n\n" + content
            else:
                plain.append({"role": role, "content": content})

        return plain

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        """Stream a single LLM call, emitting toolUse or text events."""
        tool_specs = tool_specs or []

        plain_messages = self._flatten_messages(
            messages, system_prompt, tool_specs or None
        )

        response = self.provider.complete_messages(plain_messages)

        tool_call = _extract_tool_call(response) if tool_specs else None

        yield {"messageStart": {"role": "assistant"}}

        if tool_call and tool_specs:
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            tool_use_id = f"tooluse_{uuid.uuid4().hex[:12]}"

            raw = response.strip()
            json_start = raw.find("{")
            preamble = raw[:json_start].strip() if json_start > 0 else ""
            if preamble:
                yield {"contentBlockStart": {"start": {}}}
                yield {"contentBlockDelta": {"delta": {"text": preamble}}}
                yield {"contentBlockStop": {}}

            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": tool_use_id, "name": tool_name}}
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
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": response}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

        yield {
            "metadata": {
                "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "metrics": {"latencyMs": 0},
            }
        }

    async def structured_output(
        self, output_model, prompt, system_prompt=None, **kwargs
    ):
        """Delegate structured output to stream with no tools."""
        async for event in self.stream(
            messages=prompt, tool_specs=None, system_prompt=system_prompt, **kwargs
        ):
            yield event


# ============ SNOWFLAKE CORTEX ============


class SnowflakeProvider(BaseLLMProvider):
    """Snowflake Cortex LLM provider using direct SQL-based completion."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        account=None,
        user=None,
        password=None,
        warehouse=None,
        database=None,
        schema=None,
        role=None,
    ):
        super().__init__(model, temperature, max_tokens)
        try:
            from llm_snowflake import SnowflakeCortexClient

            self._client = SnowflakeCortexClient(
                account=account,
                user=user,
                password=password,
                warehouse=warehouse,
                database=database,
                schema=schema,
                role=role,
            )
        except ImportError:
            raise ImportError("llm_snowflake.py not found in the same directory.")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a single completion request to Snowflake Cortex."""
        model_name = (
            self.model.split("/", 1)[1]
            if self.model.startswith("snowflake/")
            else self.model
        )
        return self._client.call(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
        )

    def complete_messages(self, messages: list) -> str:
        """Flatten multi-turn conversation into a single Cortex call."""
        system_parts, conv_parts = [], []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                conv_parts.append(f"Assistant: {content}")
            else:
                conv_parts.append(f"User: {content}")
        return self.complete(
            user_prompt="\n\n".join(conv_parts),
            system_prompt="\n\n".join(system_parts) or None,
        )

    def to_strands_model(self):
        """Create a SnowflakeStrandsModel adapter."""
        from snowflake_model_provider import SnowflakeStrandsModel

        model_name = (
            self.model.split("/", 1)[1]
            if self.model.startswith("snowflake/")
            else self.model
        )
        return SnowflakeStrandsModel(
            llm_client=self._client,
            model_id=model_name,
            temperature=self.temperature,
        )

    def __del__(self):
        try:
            if hasattr(self, "_client"):
                self._client.close()
        except Exception:
            pass

    @staticmethod
    def available_models():
        """Return supported Snowflake Cortex model identifiers."""
        return [
            "claude-3-5-sonnet",
            "openai-gpt-4.1",
            "openai-gpt-4.1-mini",
            "openai-gpt-4o",
            "mistral-large2",
            "llama3-70b",
            "deepseek-r1",
            "snowflake-arctic",
            "gemma-7b",
        ]


# ============ OLLAMA ============


class OllamaProvider(BaseLLMProvider):
    """Local Ollama inference provider with optional native tool support."""

    NATIVE_TOOL_MODELS = {
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "qwen2.5",
        "qwen2.5-coder",
        "mistral",
    }

    def __init__(
        self,
        model: str = "deepseek-r1:14b",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        base_url=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def _has_native_tools(self):
        """Check if the current model supports native tool calling."""
        base = self.model.split(":")[0].lower()
        return any(base.startswith(m) for m in self.NATIVE_TOOL_MODELS)

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the local Ollama instance."""
        try:
            import requests

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            return f"Ollama error: {e}"

    def _make_strands_model(self):
        """Create a native Strands OllamaModel if the model supports tools."""
        if self._has_native_tools():
            from strands.models import OllamaModel

            return OllamaModel(model_id=self.model)
        raise NotImplementedError

    @staticmethod
    def available_models():
        """Query the local Ollama instance for available models."""
        try:
            import requests

            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            r = requests.get(f"{base}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return ["deepseek-r1:14b", "qwen2.5-coder:32b", "llama3.2:3b"]


# ============ LITELLM ============


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM universal provider with native tool-calling support."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
        api_base=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self.api_base = api_base

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request via LiteLLM."""
        try:
            import litellm

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            kw = {
                "model": self.model,
                "messages": msgs,
                "temperature": self.temperature,
            }
            if self.api_key:
                kw["api_key"] = self.api_key
            if self.api_base:
                kw["api_base"] = self.api_base
            return litellm.completion(**kw).choices[0].message.content
        except Exception as e:
            return f"LiteLLM error: {e}"

    def _make_strands_model(self):
        """Create a native Strands LiteLLMModel adapter."""
        from strands.models import LiteLLMModel

        kw = {"model_id": self.model}
        ca = {}
        params = {}
        if self.api_key:
            ca["api_key"] = self.api_key
        if self.api_base:
            ca["api_base"] = self.api_base
        if self.temperature is not None:
            params["temperature"] = self.temperature
        # Prevent indefinite hangs on slow model responses
        ca.setdefault("timeout", 180)
        if ca:
            kw["client_args"] = ca
        if params:
            kw["params"] = params
        return LiteLLMModel(**kw)


# ============ OPENROUTER ============


class OpenRouterProvider(LiteLLMProvider):
    """OpenRouter API provider, routing through LiteLLM."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
        api_base=None,
    ):
        normalized_model = (
            model if model.startswith("openrouter/") else f"openrouter/{model}"
        )
        super().__init__(
            model=normalized_model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            api_base=api_base
            or os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
        )

    @staticmethod
    def available_models():
        """Return commonly available OpenRouter model identifiers."""
        return [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
        ]


# ============ OPENAI ============


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and Azure OpenAI provider."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
        azure: bool = False,
        azure_endpoint=None,
        azure_deployment=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.azure = azure
        self._api_key = api_key or (
            os.getenv("AZURE_OPENAI_API_KEY") if azure else os.getenv("OPENAI_API_KEY")
        )
        self._azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._azure_deployment = azure_deployment or model

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to OpenAI or Azure OpenAI."""
        try:
            from openai import OpenAI, AzureOpenAI

            if self.azure:
                client = AzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=self._azure_endpoint,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                )
                m = self._azure_deployment
            else:
                client = OpenAI(
                    api_key=self._api_key, organization=os.getenv("OPENAI_ORG_ID")
                )
                m = self.model
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            return (
                client.chat.completions.create(
                    model=m,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            return f"OpenAI error: {e}"

    def _make_strands_model(self):
        """Create a native Strands LiteLLMModel adapter for OpenAI."""
        from strands.models import LiteLLMModel

        return LiteLLMModel(
            model_id=f"openai/{self.model}",
            client_args={"api_key": self._api_key} if self._api_key else {},
        )

    @staticmethod
    def available_models():
        """Return supported OpenAI model identifiers."""
        return ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini"]


# ============ ANTHROPIC ============


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str = "claude-opus-4-5",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Anthropic API."""
        try:
            import anthropic

            kw = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": user_prompt}],
            }
            if system_prompt:
                kw["system"] = system_prompt
            return (
                anthropic.Anthropic(api_key=self._api_key)
                .messages.create(**kw)
                .content[0]
                .text
            )
        except Exception as e:
            return f"Anthropic error: {e}"

    def _make_strands_model(self):
        """Create a native Strands LiteLLMModel adapter for Anthropic."""
        from strands.models import LiteLLMModel

        return LiteLLMModel(
            model_id=f"anthropic/{self.model}",
            client_args={"api_key": self._api_key} if self._api_key else {},
        )

    @staticmethod
    def available_models():
        """Return supported Anthropic model identifiers."""
        return [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
        ]


# ============ GEMINI ============


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            m = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature, max_output_tokens=self.max_tokens
                ),
            )
            return m.generate_content(user_prompt).text
        except Exception as e:
            return f"Gemini error: {e}"

    def _make_strands_model(self):
        """Create a native Strands LiteLLMModel adapter for Gemini."""
        from strands.models import LiteLLMModel

        return LiteLLMModel(
            model_id=f"gemini/{self.model}",
            client_args={"api_key": self._api_key} if self._api_key else {},
        )

    @staticmethod
    def available_models():
        """Return supported Google Gemini model identifiers."""
        return ["gemini-2.0-flash", "gemini-2.5-pro", "gemini-1.5-pro"]


# ============ BEDROCK ============


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider with native Strands tool support."""

    def __init__(
        self,
        model: str = "us.amazon.nova-lite-v1:0",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        region=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to AWS Bedrock."""
        try:
            import boto3

            client = boto3.client("bedrock-runtime", region_name=self.region)
            kw = {
                "modelId": self.model,
                "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
                "inferenceConfig": {
                    "temperature": self.temperature,
                    "maxTokens": self.max_tokens,
                },
            }
            if system_prompt:
                kw["system"] = [{"text": system_prompt}]
            return client.converse(**kw)["output"]["message"]["content"][0]["text"]
        except Exception as e:
            return f"Bedrock error: {e}"

    def _make_strands_model(self):
        """Create a native Strands BedrockModel adapter."""
        from strands.models import BedrockModel

        return BedrockModel(model_id=self.model, region_name=self.region)

    @staticmethod
    def available_models():
        """Return supported AWS Bedrock model identifiers."""
        return [
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-pro-v1:0",
            "us.anthropic.claude-opus-4-5-20250514-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250514-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "us.deepseek.r1-v1:0",
        ]


# ============ ADDITIONAL PROVIDERS ============


class MistralProvider(BaseLLMProvider):
    """Mistral AI API provider."""

    def __init__(
        self,
        model: str = "mistral-large-latest",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Mistral API."""
        try:
            from mistralai import Mistral

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            return (
                Mistral(api_key=self._api_key)
                .chat.complete(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            return f"Mistral error: {e}"

    def _make_strands_model(self):
        """Create a native Strands LiteLLMModel adapter for Mistral."""
        from strands.models import LiteLLMModel

        return LiteLLMModel(
            model_id=f"mistral/{self.model}",
            client_args={"api_key": self._api_key} if self._api_key else {},
        )

    @staticmethod
    def available_models():
        """Return supported Mistral model identifiers."""
        return ["mistral-large-latest", "codestral-latest", "open-mixtral-8x22b"]


class GroqProvider(BaseLLMProvider):
    """Groq inference API provider for ultra-fast inference."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or os.getenv("GROQ_API_KEY")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Groq API."""
        try:
            from groq import Groq

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            return (
                Groq(api_key=self._api_key)
                .chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            return f"Groq error: {e}"

    def _make_strands_model(self):
        """Create a native Strands LiteLLMModel adapter for Groq."""
        from strands.models import LiteLLMModel

        return LiteLLMModel(
            model_id=f"groq/{self.model}",
            client_args={"api_key": self._api_key} if self._api_key else {},
        )

    @staticmethod
    def available_models():
        """Return supported Groq model identifiers."""
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
        ]


class CohereProvider(BaseLLMProvider):
    """Cohere API provider."""

    def __init__(
        self,
        model: str = "command-r-plus-08-2024",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or os.getenv("COHERE_API_KEY")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Cohere API."""
        try:
            import cohere

            kw = {
                "model": self.model,
                "message": user_prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if system_prompt:
                kw["preamble"] = system_prompt
            return cohere.Client(api_key=self._api_key).chat(**kw).text
        except Exception as e:
            return f"Cohere error: {e}"


class TogetherProvider(BaseLLMProvider):
    """Together AI API provider."""

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = api_key or os.getenv("TOGETHER_API_KEY")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Together AI API."""
        try:
            from together import Together

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            return (
                Together(api_key=self._api_key)
                .chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            return f"Together AI error: {e}"


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Inference API provider."""

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self._api_key = (
            api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
        )

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to the Hugging Face Inference API."""
        try:
            from huggingface_hub import InferenceClient

            prompt = (
                f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
                if system_prompt
                else user_prompt
            )
            return InferenceClient(
                model=self.model, token=self._api_key
            ).text_generation(
                prompt,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),
            )
        except Exception as e:
            return f"HuggingFace error: {e}"


class CustomHTTPProvider(BaseLLMProvider):
    """Generic OpenAI-compatible HTTP endpoint provider."""

    def __init__(
        self,
        model: str = "local-model",
        base_url=None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        api_key=None,
    ):
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url or os.getenv(
            "CUSTOM_LLM_URL", "http://localhost:8000/v1"
        )
        self._api_key = api_key or os.getenv("CUSTOM_LLM_API_KEY", "dummy")

    def complete(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Send a completion request to a custom OpenAI-compatible endpoint."""
        try:
            from openai import OpenAI

            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": user_prompt})
            return (
                OpenAI(api_key=self._api_key, base_url=self.base_url)
                .chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                .choices[0]
                .message.content
            )
        except Exception as e:
            return f"Custom HTTP error: {e}"


# ============ ROUTER ============


class LLMRouter:
    """Factory router that instantiates the appropriate LLM provider.

    Supports explicit provider selection, environment-based auto-detection,
    and model-string inference.
    """

    PROVIDER_MAP = {
        "snowflake": SnowflakeProvider,
        "ollama": OllamaProvider,
        "litellm": LiteLLMProvider,
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "azure": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "gemini": GeminiProvider,
        "google": GeminiProvider,
        "bedrock": BedrockProvider,
        "aws": BedrockProvider,
        "mistral": MistralProvider,
        "groq": GroqProvider,
        "cohere": CohereProvider,
        "together": TogetherProvider,
        "together_ai": TogetherProvider,
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,
        "custom": CustomHTTPProvider,
    }

    DEFAULTS = {
        "snowflake": "claude-3-5-sonnet",
        "ollama": "deepseek-r1:14b",
        "litellm": "openai/gpt-4o",
        "openrouter": "openai/gpt-4o-mini",
        "openai": "gpt-4o",
        "azure": "gpt-4o",
        "anthropic": "claude-opus-4-5",
        "claude": "claude-opus-4-5",
        "gemini": "gemini-2.0-flash",
        "google": "gemini-2.0-flash",
        "bedrock": "us.amazon.nova-lite-v1:0",
        "aws": "us.amazon.nova-lite-v1:0",
        "mistral": "mistral-large-latest",
        "groq": "llama-3.3-70b-versatile",
        "cohere": "command-r-plus-08-2024",
        "together": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
        "huggingface": "mistralai/Mistral-7B-Instruct-v0.3",
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
        "custom": "local-model",
    }

    MODEL_PREFIX_MAP = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
        "openrouter/": "openrouter",
        "claude-": "anthropic",
        "gemini-": "gemini",
        "llama": "ollama",
        "deepseek": "ollama",
        "qwen": "ollama",
        "phi": "ollama",
        "openai-gpt": "snowflake",
        "amazon.": "bedrock",
        "us.amazon.": "bedrock",
        "us.anthropic.": "bedrock",
        "us.meta.": "bedrock",
        "command-": "cohere",
        "meta-llama/": "together",
    }

    @classmethod
    def create(
        cls,
        provider: str,
        model=None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Instantiate a provider by name with the given configuration."""
        p = provider.lower()
        if p not in cls.PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {list(cls.PROVIDER_MAP)}"
            )
        if model is None:
            model = cls.DEFAULTS.get(p, "default")
        klass = cls.PROVIDER_MAP[p]
        if p == "azure":
            return klass(
                model=model,
                azure=True,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        return klass(
            model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    @classmethod
    def from_env(cls, temperature: float = 0.0) -> BaseLLMProvider:
        """Auto-detect the best available provider from environment variables."""
        provider_hint = os.getenv("SWARM_PROVIDER", "").strip().lower()
        if provider_hint:
            if provider_hint not in cls.PROVIDER_MAP:
                raise ValueError(
                    f"Unknown SWARM_PROVIDER '{provider_hint}'. Available: {list(cls.PROVIDER_MAP)}"
                )
            model_hint = os.getenv("SWARM_MODEL", "").strip() or None
            print(f"[LLMRouter] SWARM_PROVIDER override: {provider_hint}")
            return cls.create(provider_hint, model=model_hint, temperature=temperature)

        m = os.getenv("SWARM_MODEL", "")
        if m:
            if "/" not in m and (
                (os.getenv("SNOWFLAKE_ACCOUNT") and os.getenv("SNOWFLAKE_USER"))
                or (os.getenv("SNOWFLAKE_ACCOUNT_ID") and os.getenv("SNOWFLAKE_JWT"))
                or os.getenv("SNOWFLAKE_API_BASE")
            ):
                print("[LLMRouter] SWARM_MODEL + Snowflake config detected")
                return cls.create("snowflake", model=m, temperature=temperature)
            return cls.from_model_string(m, temperature)

        has_snowflake = (
            (os.getenv("SNOWFLAKE_ACCOUNT") and os.getenv("SNOWFLAKE_USER"))
            or (os.getenv("SNOWFLAKE_ACCOUNT_ID") and os.getenv("SNOWFLAKE_JWT"))
            or os.getenv("SNOWFLAKE_API_BASE")
        )
        if has_snowflake:
            print("[LLMRouter] Auto-detected: snowflake")
            return cls.create("snowflake", temperature=temperature)

        for k1, k2, p in [
            ("ANTHROPIC_API_KEY", None, "anthropic"),
            ("OPENROUTER_API_KEY", None, "openrouter"),
            ("OPENAI_API_KEY", None, "openai"),
            ("GOOGLE_API_KEY", None, "gemini"),
            ("GROQ_API_KEY", None, "groq"),
            ("MISTRAL_API_KEY", None, "mistral"),
        ]:
            if os.getenv(k1) and (k2 is None or os.getenv(k2)):
                print(f"[LLMRouter] Auto-detected: {p}")
                return cls.create(p, temperature=temperature)
        print("[LLMRouter] Falling back to Ollama")
        return cls.create("ollama", temperature=temperature)

    @classmethod
    def from_model_string(cls, s: str, temperature: float = 0.0) -> BaseLLMProvider:
        """Infer the provider from a model identifier string."""
        if "/" in s:
            p, model = s.split("/", 1)
            p = p.lower()
            if p in cls.PROVIDER_MAP:
                return cls.create(p, model=model, temperature=temperature)
            return cls.create("litellm", model=s, temperature=temperature)
        for prefix, prov in cls.MODEL_PREFIX_MAP.items():
            if s.startswith(prefix):
                return cls.create(prov, model=s, temperature=temperature)
        print(f"[LLMRouter] Unknown '{s}' — trying LiteLLM")
        return cls.create("litellm", model=s, temperature=temperature)

    @classmethod
    def list_all(cls) -> dict:
        """Return available models for all known providers."""
        return {
            "snowflake": SnowflakeProvider.available_models(),
            "ollama": OllamaProvider.available_models(),
            "openrouter": OpenRouterProvider.available_models(),
            "openai": OpenAIProvider.available_models(),
            "anthropic": AnthropicProvider.available_models(),
            "gemini": GeminiProvider.available_models(),
            "bedrock": BedrockProvider.available_models(),
            "mistral": MistralProvider.available_models(),
            "groq": GroqProvider.available_models(),
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        for prov, ms in LLMRouter.list_all().items():
            print(f"\n{prov}:")
            for m in ms:
                print(f"  - {m}")
    else:
        llm = (
            LLMRouter.create(args.provider, model=args.model)
            if args.provider
            else LLMRouter.from_env()
        )
        print(f"Provider: {type(llm).__name__}  Model: {llm.model}")
        print(f"Strands model: {type(llm.to_strands_model()).__name__}")
        print(llm.complete("Say hello in 5 words.", "You are concise."))
