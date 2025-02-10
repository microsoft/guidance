from dataclasses import dataclass, field
from typing import Union
from uuid import UUID, uuid4

from PIL.ImageFile import ImageFile as PILImageFile

from guidance._grammar import GrammarFunction, RawFunction

# TODO
Node = Union[str, GrammarFunction, RawFunction, "ImageBlob"]
ContentChunk = Union[str, "ImageBlob"]


@dataclass(frozen=True, slots=True)
class RoleStart:
    role: str
    id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True, slots=True)
class RoleEnd:
    id: UUID


@dataclass(frozen=True, slots=True)
class ImageBlob:
    image: PILImageFile

    def __add__(self, other: Node) -> RawFunction:
        # Bit of a hack for now to make sure we are handled by the model directly
        # and can't be used inside a select statement
        def __add__(model):
            model += self
            model += other
            return model

        return RawFunction(__add__, [], {})


MessageChunk = Union[ContentChunk, RoleStart, RoleEnd]
