from typing import Union

from .base import ChatState, CompletionState, RoleEnd, RoleStart
from .openai import OpenAIState
from .transformers import (
    BaseTransformersChatState,
    Llama3TransformersState,
    TransformersUnstructuredState,
)

State = Union[ChatState, CompletionState]

__all__ = [
    "State",
    "ChatState",
    "CompletionState",
    "RoleEnd",
    "RoleStart",
    "OpenAIState",
    "TransformersUnstructuredState",
    "Llama3TransformersState",
    "BaseTransformersChatState",
]
