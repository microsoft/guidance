import re
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Iterator, Optional

from typing_extensions import Self

from guidance._grammar import Null, RawFunction, _call_pool, _tag_pattern

from .ast import MessageChunk, Node, RoleEnd, RoleStart
from .client import Client

_active_role: ContextVar[Optional["RoleStart"]] = ContextVar("active_role", default=None)


class Model:
    def __init__(
        self,
        client: Client,
    ) -> None:
        self.client = client
        self._state = client.initial_state()
        self._active_role: Optional["RoleStart"] = None

    def __add__(self, other: Node) -> None:
        self = self.copy()
        self._apply_role_changes()
        if isinstance(other, str):
            if other == "":
                return self
            other = extract_embedded_nodes(other)
        if isinstance(other, RawFunction):
            return other(self)
        self._apply_node(other)
        return self

    def _apply_node(self, node: Node) -> None:
        for chunk in self.client.run(self._state, node):
            self._apply_chunk(chunk)

    def _apply_chunk(self, chunk: MessageChunk) -> None:
        self._state.apply_chunk(chunk)

    @contextmanager
    def role(self, role: str) -> Iterator[None]:
        # _apply_chunk will raise an exception via _api_state.apply_chunk if roles are not supported
        role_start = RoleStart(role)
        token = _active_role.set(role_start)
        try:
            yield
        finally:
            _active_role.reset(token)

    def _apply_role_changes(self) -> None:
        active_role = _active_role.get()
        if self._active_role != active_role:
            if self._active_role is not None:
                self._apply_chunk(RoleEnd(self._active_role.id))
            if active_role is not None:
                self._apply_chunk(active_role)
            self._active_role = active_role

    def system(self) -> AbstractContextManager[None]:
        return self.role("system")

    def user(self) -> AbstractContextManager[None]:
        return self.role("user")

    def assistant(self) -> AbstractContextManager[None]:
        return self.role("assistant")

    def __str__(self) -> str:
        return self.client.format_state(self._state)

    def copy(self) -> Self:
        obj = object.__new__(self.__class__)
        obj.client = self.client
        obj._active_role = self._active_role
        obj._state = deepcopy(self._state)
        return obj


def extract_embedded_nodes(value: str) -> Node:
    parts: list[str] = re.split(_tag_pattern, value)

    if len(parts) == 1:
        return value

    is_id = False
    grammar = Null()
    for part in parts:
        if is_id:
            call = _call_pool[part]
            grammar += call
        else:
            grammar += part
        is_id = not is_id
    return grammar
