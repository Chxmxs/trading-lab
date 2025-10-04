"""
LLM provider abstraction for the AI companion.

This module defines a thin wrapper around different language model
providers (currently OpenAI and Azure OpenAI) to enable easy switching
between models.  The abstraction reads configuration from environment
variables so that no secrets are hard coded in the repository.  It
also exposes a method to call the model with a set of function/tool
definitions, enabling the model to return structured results as part
of a patch proposal.

Refer to OpenAI's documentation for details on function calling and
tool usage.  Function calling allows models like GPT‑4 to return
JSON arguments to call external functions, acting as an interface
between natural language and code.  To use the OpenAI
Python library, you must set the ``OPENAI_API_KEY`` environment
variable or manually assign ``openai.api_key``.

Example
-------

>>> from companion.ai_core import AIClient
>>> client = AIClient()
>>> tools = [
...     {
...         "type": "function",
...         "function": {
...             "name": "propose_patch",
...             "description": "Propose a configuration change",
...             "parameters": {
...                 "type": "object",
...                 "properties": {
...                     "risk": {"type": "string"},
...                     "diff": {"type": "string"},
...                 },
...                 "required": ["risk", "diff"],
...             },
...         },
...     },
... ]
>>> prompt = "The run failed because the SOPR file is missing. Suggest a fix."
>>> response = client.chat(prompt, tools=tools)
>>> print(response)

The above example is illustrative; the actual implementation should
include proper error handling and tool invocation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # OpenAI library is optional; loaded based on provider

@dataclass
class AIClientConfig:
    """Configuration for the AI client.

    Attributes
    ----------
    provider: str
        Name of the LLM provider (e.g. ``"openai"`` or ``"azure"``).  Defaults
        to ``"openai"``.
    model: str
        Name of the model to use.  Defaults to ``"gpt-4o"`` which supports
        function calling.  Users can override this via the ``LLM_MODEL``
        environment variable.
    api_key: Optional[str]
        Secret API key.  If ``None``, the client will look for
        ``OPENAI_API_KEY`` or ``LLM_API_KEY`` in the environment.
    api_base: Optional[str]
        Custom API base URL (e.g. for Azure).  Optional.
    """

    provider: str = os.environ.get("LLM_PROVIDER", "openai")
    model: str = os.environ.get("LLM_MODEL", "gpt-4o")
    api_key: Optional[str] = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base: Optional[str] = os.environ.get("OPENAI_API_BASE")

class AIClient:
    """Abstracts interactions with an LLM provider.

    The client currently supports the OpenAI API and can be extended to
    support additional providers by adding branches in :meth:`_send_request`.
    All network calls should be made through this class so that they can
    be mocked during testing.  In production use, you would catch
    exceptions here and implement retries/backoff as appropriate.
    """

    def __init__(self, config: Optional[AIClientConfig] = None) -> None:
        self.config = config or AIClientConfig()
        # Validate configuration
        if self.config.provider == "openai" and openai is None:
            raise RuntimeError(
                "openai package is not installed. Run `pip install openai` to use this provider."
            )
        # Configure OpenAI if used
        if self.config.provider == "openai" and openai is not None:
            if self.config.api_key:
                openai.api_key = self.config.api_key
            if self.config.api_base:
                # Support custom endpoints (e.g. Azure)
                openai.api_base = self.config.api_base

    def chat(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Send a prompt to the LLM and optionally provide tool definitions.

        Parameters
        ----------
        prompt: str
            The natural language query to send to the model.
        tools: Optional[List[Dict[str, Any]]], optional
            A list of tool (function) specifications as defined in the OpenAI
            API.  When provided, the model may choose to call a tool by
            returning a JSON structure with arguments.
        **kwargs: Any
            Additional keyword arguments forwarded to the underlying API call.

        Returns
        -------
        Dict[str, Any]
            The parsed response from the LLM, including any tool calls.
        """
        if self.config.provider == "openai":
            return self._chat_openai(prompt, tools=tools, **kwargs)
        raise NotImplementedError(f"Provider {self.config.provider} not supported yet")

    def _chat_openai(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> Dict[str, Any]:
        """Send a chat request using the OpenAI API.

        This helper constructs the message list and calls the chat
        completions endpoint.  It does not perform any retries or
        streaming; callers should wrap this in their own retry logic if
        necessary.  The return value is the first choice from the
        response converted to a plain dictionary.
        """
        if openai is None:
            raise RuntimeError("openai package is not available. Please install it.")
        messages = [
            {"role": "system", "content": "You are an assistant helping to maintain a trading bot."},
            {"role": "user", "content": prompt},
        ]
        params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            params["tools"] = tools
            # In OpenAI API, tool_choice="auto" allows the model to decide when
            # to call a function.
            params["tool_choice"] = "auto"
        params.update(kwargs)
        # Placeholder for API call; not executed in skeleton
        # response = openai.chat.completions.create(**params)
        # For a skeleton, we simply return an empty structure.
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    }
                }
            ]
        }

__all__ = ["AIClient", "AIClientConfig"]
