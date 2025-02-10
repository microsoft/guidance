from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Generic,
    MutableSequence,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
)

from typing_extensions import Self, assert_never

from ..ast import ContentChunk, ImageBlob, MessageChunk, RoleEnd, RoleStart

# Return type of BaseState.get_state
R = TypeVar("R", covariant=True)


class BaseState(Generic[R], ABC):
    def __init__(self) -> None:
        self.chunks: MutableSequence[MessageChunk] = []

    @classmethod
    def from_chunks(cls, chunks: Sequence[MessageChunk]) -> Self:
        self = cls()
        for chunk in chunks:
            self.apply_chunk(chunk)
        return self

    def apply_chunk(self, chunk: MessageChunk) -> None:
        self.chunks.append(chunk)
        match chunk:
            case RoleStart(_, _) as role_start:
                self.apply_role_start(role_start)
            case RoleEnd(_) as role_end:
                self.apply_role_end(role_end)
            case _:
                self.apply_content_chunk(chunk)

    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        match chunk:
            case str(text):
                self.apply_text(text)
            case ImageBlob(_) as image:
                self.apply_image(image)
            case _:
                if TYPE_CHECKING:
                    assert_never(chunk)
                raise NotImplementedError(f"Chunk type {type(chunk)} not supported")

    @abstractmethod
    def get_state(self) -> R:
        pass

    @abstractmethod
    def apply_role_start(self, role_start: RoleStart) -> None:
        pass

    @abstractmethod
    def apply_role_end(self, role_end: RoleEnd) -> None:
        pass

    @abstractmethod
    def apply_text(self, text: str) -> None:
        pass

    def apply_image(self, image: ImageBlob) -> None:
        # TODO: raise custom exception so we can catch it and raise a better error
        # where we have the model's name, etc.
        raise TypeError(f"Image blobs not supported by {self.__class__.__name__}")


class BaseCompletionStateObj(TypedDict):
    prompt: str


class CompletionState(BaseState[BaseCompletionStateObj]):
    def __init__(self) -> None:
        super().__init__()
        self.prompt = ""

    def apply_role_start(self, role_start: RoleStart) -> None:
        raise TypeError("Role blocks not supported for completion models")

    def apply_role_end(self, role_end: RoleEnd) -> None:
        raise TypeError("Role blocks not supported for completion models")

    def apply_text(self, text: str) -> None:
        self.prompt += text

    def get_state(self) -> BaseCompletionStateObj:
        return {"prompt": self.prompt}


class BaseChatMessage(TypedDict):
    role: str


# Message type of BaseChatState
M = TypeVar("M", bound=BaseChatMessage, covariant=True)


class BaseChatStateObj(TypedDict, Generic[M]):
    messages: Sequence[M]
    prefill: Optional[M]
    active_role: Optional[str]


class ChatState(Generic[M], BaseState[BaseChatStateObj[M]], ABC):
    def __init__(self) -> None:
        super().__init__()
        self.active_role: Optional[RoleStart] = None
        self.messages: MutableSequence[M] = []

    def apply_role_start(self, role_start: RoleStart) -> None:
        if self.active_role is not None:
            raise ValueError(
                f"Cannot open role {role_start.role!r}: {self.active_role.role!r} is already open."
            )
        self.active_role = role_start

    def apply_role_end(self, role_end: RoleEnd) -> None:
        if self.active_role is None:
            raise ValueError("Cannot close role without active role")
        if self.active_role.id != role_end.id:
            raise ValueError("RoleEnd does not match active role")
        active_message = self.get_active_message()
        if active_message is not None:
            self.messages.append(active_message)
        self.reset_active_message()

    def apply_content_chunk(self, chunk: ContentChunk) -> None:
        if self.active_role is None:
            raise ValueError(
                "Cannot add to chat model outside of a role block (use model.system(), model.user(), or model.assistant() context managers)"
            )
        super().apply_content_chunk(chunk)

    def reset_active_message(self) -> None:
        self.active_role = None

    def get_state(self) -> BaseChatStateObj[M]:
        return {
            "messages": self.messages,
            "prefill": self.get_active_message(),
            "active_role": self.get_active_role(),
        }

    def get_active_role(self) -> Optional[str]:
        if self.active_role is None:
            return None
        return self.active_role.role

    @abstractmethod
    def get_active_message(self) -> Optional[M]:
        pass
