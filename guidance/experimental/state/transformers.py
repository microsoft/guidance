import re
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, Sequence, TypedDict, TypeVar

from ..ast import ImageBlob
from .base import ChatState

# Type of content in a message
TC = TypeVar("TC")


class TransformersMessage(TypedDict, Generic[TC]):
    role: str
    content: TC


class TransformersChatObj(TypedDict, Generic[TC]):
    messages: Sequence[TransformersMessage[TC]]
    prefill: Optional[TransformersMessage[TC]]
    active_role: Optional[str]
    images: list[Any]
    audio: list[Any]
    videos: list[Any]


class BaseTransformersChatState(ChatState[TransformersMessage[TC]], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.content: TC = self._default_content_factory()
        self.images: list[Any] = []
        self.audio: list[Any] = []
        self.videos: list[Any] = []

    @classmethod
    def from_model_id(cls, model_id: str) -> "BaseTransformersChatState":
        if "Phi-3-vision" in model_id:
            return Phi3VisionState()
        if re.search("Llama-3.*-Vision", model_id):
            return Llama3TransformersState()
        # Fallback to unstructured (string) state
        return TransformersUnstructuredState()

    def get_active_message(self) -> Optional[TransformersMessage[TC]]:
        if self.active_role is None:
            return None

        return TransformersMessage({"role": self.active_role.role, "content": self.content})

    def get_state(self) -> TransformersChatObj[TC]:
        return {
            "messages": self.messages,
            "prefill": self.get_active_message(),
            "active_role": self.get_active_role(),
            "images": self.images,
            "audio": self.audio,
            "videos": self.videos,
        }

    def reset_active_message(self) -> None:
        super().reset_active_message()
        # Don't delete images, audio, or videos, as they are not part of the message
        self.content = self._default_content_factory()

    @abstractmethod
    def _default_content_factory(self) -> TC:
        pass


class TransformersStructuredState(BaseTransformersChatState[list[dict]]):
    def _default_content_factory(self) -> list[dict]:
        return []


class TransformersUnstructuredState(BaseTransformersChatState[str]):
    def _default_content_factory(self) -> str:
        return ""

    def apply_text(self, text: str) -> None:
        self.content += text


class Llama3TransformersState(TransformersStructuredState):
    def apply_text(self, text: str) -> None:
        self.content.append({"type": "text", "text": text})

    def apply_image(self, image: ImageBlob) -> None:
        self.images.append(image.image)
        self.content.append({"type": "image", "image": "image"})


class Phi3VisionState(TransformersUnstructuredState):
    def apply_image(self, image: ImageBlob) -> None:
        pil_image = image.image
        if pil_image in self.images:
            self.content += f"<|image_{self.images.index(pil_image) + 1}|>"
        else:
            self.images.append(image.image)
            self.content += f"<|image_{len(self.images)}|>"
