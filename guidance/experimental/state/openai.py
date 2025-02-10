from base64 import b64encode
from io import BytesIO
from typing import Optional, TypedDict, Union

from ..ast import ImageBlob
from .base import ChatState


class OpenAIContentMessage(TypedDict):
    role: str
    content: list[dict]


class OpenAIAudioMessage(TypedDict):
    role: str
    audio: dict


OpenAIMessage = Union[OpenAIContentMessage, OpenAIAudioMessage]


class OpenAIState(ChatState[OpenAIMessage]):
    def __init__(self) -> None:
        super().__init__()
        self.content: list[dict] = []
        self.audio: Optional[dict] = None

    @classmethod
    def from_openai_model(cls, model: str) -> "OpenAIState":
        if "audio-preview" in model:
            return OpenAIAudioState()
        if model.startswith("gpt-4o") or model.startswith("o1"):
            return OpenAIImageState()
        else:
            return OpenAIState()

    def get_active_message(self) -> Optional[OpenAIMessage]:
        if self.active_role is None:
            return None

        if self.audio:
            if self.content:
                raise ValueError("Expected either content or audio in OpenAI message, not both")
            return OpenAIAudioMessage({"role": self.active_role.role, "audio": self.audio})

        if self.content:
            return OpenAIContentMessage({"role": self.active_role.role, "content": self.content})

        return None

    def reset_active_message(self) -> None:
        super().reset_active_message()
        self.content = []
        self.audio = None

    def apply_text(self, text: str) -> None:
        self.content.append({"type": "text", "text": text})


class OpenAIImageState(OpenAIState):
    def apply_image(self, image: ImageBlob) -> None:
        format = image.image.format
        if format is None:
            raise ValueError(f"Cannot upload image with unknown format: {image.image}")

        with BytesIO() as buffer:
            image.image.save(buffer, format=format)
            b64_image = b64encode(buffer.getvalue()).decode("utf-8")

        mime_type = f"image/{format.lower()}"
        self.content.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}}
        )


class OpenAIAudioState(OpenAIState):
    def __init__(self) -> None:
        raise NotImplementedError("OpenAI audio not yet implemented")
