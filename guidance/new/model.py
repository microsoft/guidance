from contextlib import AbstractContextManager, contextmanager
from typing import Iterator

from typing_extensions import Self

from .ast import MessageChunk, Node, RoleEnd, RoleStart
from .client import Client
from .state import APIState, InternalState


class Model:
    def __init__(
        self,
        client: Client,
        api_state: APIState,
    ) -> None:
        self.client = client
        self._api_state = api_state
        self._internal_state = InternalState()

    def __iadd__(self, other: Node) -> Self:
        self._apply_node(other)
        return self

    def __add__(self, other: Node) -> None:
        raise TypeError("Use += to add nodes")

    def _apply_node(self, node: Node) -> None:
        for chunk in self.client.run(self._api_state, node):
            self._apply_chunk(chunk)

    def _apply_chunk(self, chunk: MessageChunk) -> None:
        # Apply to _api_state first, so that it can raise an exception if the chunk is not supported
        self._api_state.apply_chunk(chunk)
        self._internal_state.apply_chunk(chunk)

    @contextmanager
    def role(self, role: str) -> Iterator[None]:
        # _apply_chunk will raise an exception via _api_state.apply_chunk if roles are not supported
        role_start = RoleStart(role)
        self._apply_chunk(role_start)
        try:
            yield
        finally:
            role_end = RoleEnd(role_start.id)
            self._apply_chunk(role_end)

    def system(self) -> AbstractContextManager[None]:
        return self.role("system")

    def user(self) -> AbstractContextManager[None]:
        return self.role("user")

    def assistant(self) -> AbstractContextManager[None]:
        return self.role("assistant")
