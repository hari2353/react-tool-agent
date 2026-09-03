"""Public API surface for the tool agent package."""

from tool_agent.agent import AgentConfig, ToolAgent
from tool_agent.llm import AzureOpenAILLM, ChatLLM, LLMError, ScriptedLLM, parse_decision
from tool_agent.models import AgentResult, AgentStep, Role, ToolCall, ToolSpec
from tool_agent.retrieval import Passage, SearchHit, TfIdfStore, default_store
from tool_agent.tools import (
    ToolError,
    ToolRegistry,
    default_registry,
    safe_eval,
    unit_convert,
)

__all__ = [
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    "AzureOpenAILLM",
    "ChatLLM",
    "LLMError",
    "Passage",
    "Role",
    "ScriptedLLM",
    "SearchHit",
    "TfIdfStore",
    "ToolAgent",
    "ToolCall",
    "ToolError",
    "ToolRegistry",
    "ToolSpec",
    "default_registry",
    "default_store",
    "parse_decision",
    "safe_eval",
    "unit_convert",
]
