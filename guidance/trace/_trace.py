# TODO(nopdive): Consider integrating token operations into trace nodes (handles token healing cleaner).
# TODO(nopdive): Benchmark (expected heap fragmentation issue). Likely need memory pooling (via rust/ctypes/Cython).
import weakref
from itertools import count
from typing import Any, Optional, Generator, Dict
import logging
from pydantic import BaseModel, Field
from guidance._schema import GenToken
from .._utils import pydantic_no_default_repr, pydantic_no_default_str, log_cleanup

logger = logging.getLogger(__name__)


class NodeAttr(BaseModel):
    """Attributes of a trace node."""

    class_name: str = ""

    def __init__(self, **kwargs):
        kwargs["class_name"] = self.__class__.__name__
        super().__init__(**kwargs)

    def __repr__(self):
        return pydantic_no_default_repr(self)

    def __str__(self):
        return pydantic_no_default_str(self)


class InputAttr(NodeAttr):
    """Input for a guidance program (i.e. literal or guidance grammar)."""

    pass


class OutputAttr(NodeAttr):
    """Output for a guidance program (i.e. text output)."""

    pass


class StatelessGuidanceInput(InputAttr):
    """Stateless guidance input (light wrapper)."""

    # NOTE(nopdive): Open to debate what we should serialize here, excluding for now.
    value: Any = Field(exclude=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class StatefulGuidanceInput(InputAttr):
    """Stateful guidance input (light wrapper)."""

    # NOTE(nopdive): Open to debate what we should serialize here, excluding for now.
    value: Any = Field(exclude=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class LiteralInput(InputAttr):
    """Text string as a literal."""

    value: str


class ImageInput(InputAttr):
    """Image input."""

    value: str


class AudioInput(InputAttr):
    """Audio input."""
    value: str


class VideoInput(InputAttr):
    """Video input."""
    value: str


class EmbeddedInput(InputAttr):
    """Text string with embedded guidance input."""

    value: str


class RoleOpenerInput(InputAttr):
    """Appears when a role is opened (i.e. user / system).

    This usually occurs as a role context and __enter__ is called.
    """

    name: Optional[str] = None
    text: Optional[str] = None
    closer_text: Optional[str] = None


class RoleCloserInput(InputAttr):
    """Appears when a role is closed (i.e. user / system).

    This usually occurs as a role context and __exit__ is called.
    """

    name: Optional[str] = None
    text: Optional[str] = None


class AudioOutput(OutputAttr):
    """Audio output."""
    value: str
    is_input: bool = False


class VideoOutput(OutputAttr):
    """Video output."""
    value: str
    is_input: bool = False


class ImageOutput(OutputAttr):
    """Image output."""

    value: str
    is_input: bool = False


class TextOutput(OutputAttr):
    """Text string."""

    value: str
    is_input: bool = False
    is_generated: bool = False
    is_force_forwarded: bool = False
    token_count: int = 0
    prob: float = 0.0
    tokens: list[GenToken] = Field(
        exclude=True, default_factory=list
    )  # use to store tokens associated with output

    def __str__(self):
        return self.value

class CaptureOutput(OutputAttr):
    """Capture variable output as a string.

    If `value` is set to None, this means it's a reset (needed for append capture group outputs).
    """

    name: str
    value: Optional[str] = None
    is_append: bool = False
    log_probs: float = 0.0

    def __str__(self):
        return f"{self.name}{'+=' if self.is_append else '='}{self.value.__str__()}"


class WeakRefList(list):
    """Weak reference list implementation that uses weakref ref objects.

    This does not override all methods for list.
    """

    def append(self, item):
        super().append(weakref.ref(item))

    def __getitem__(self, index):
        ref = super().__getitem__(index)
        obj = ref()
        if obj is None:
            raise ReferenceError("The referenced object has been garbage collected")
        return obj

    def __iter__(self):
        return (ref() for ref in super().__iter__() if ref() is not None)

    def remove(self, item):
        ref_to_remove = None
        for ref in super().__iter__():
            obj = ref()
            if obj is item:
                ref_to_remove = ref
                break
        if ref_to_remove:
            super().remove(ref_to_remove)


def _cleanup(log_msg: str):
    log_cleanup(log_msg)


class TraceNode(BaseModel):
    """Trace node which associates inputs and outputs of a guidance program."""

    identifier: int = Field(default_factory=count().__next__)
    parent: Optional["TraceNode"] = None
    children: list["TraceNode"] = Field(default_factory=WeakRefList)
    input: Optional[InputAttr] = None
    output: Optional[OutputAttr] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        weakref.finalize(self, _cleanup, f"trace({id(self)}):{self.identifier}")

    def add_child(self, child: "TraceNode") -> None:
        """Add a child node to the trace node.

        Args:
            child: The child node to add.
        """
        child.parent = self
        self.children.append(child)

    def remove_child(self, child: "TraceNode") -> None:
        """Remove a child node from the trace node.

        Args:
            child: The child node to remove.
        """
        child.parent = None
        self.children.remove(child)

    def ancestors(self) -> Generator["TraceNode", None, None]:
        """Yields ancestor trace nodes ending with root.

        Yields:
            Ancestor trace nodes.
        """
        node = self
        while node.parent is not None:
            yield node.parent
            node = node.parent

    def path(self) -> Generator["TraceNode", None, None]:
        """Yields path of from root to self inclusively.

        Yields:
            Trace nodes from root to self.
        """
        yield from reversed(list(self.ancestors()))
        yield self

    def root(self) -> "TraceNode":
        """Returns root by traversing parents of self.

        Returns:
            Root of tree self is part of.
        """
        if self.parent is None:
            root = self
        else:
            *_, root = self.ancestors()
        return root

    def traverse(self, bfs: bool = True):
        """Traverse the trace nodes starting with self.

        Args:
            bfs: Use breadth-first-search, otherwise depth-first-search.

        Yields:
            Trace nodes in traversal order.
        """
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node

            if bfs:
                queue.extend(node.children)
            else:
                # NOTE(nopdive): Analogous to extend but at front of list.
                queue[0:0] = node.children

    def __repr__(self):
        return f"{self.identifier}:{self.input!r}:{self.output!r}"

    def __hash__(self):
        return hash(self.identifier)


class TraceHandler(BaseModel):
    """Trace handler that will own a tree of trace nodes.

    This will primarily be owned by a model's engine.
    Each guidance model corresponds to one trace node.
    All guidance models emitted from an engine is to be included as their own paths within the tree.

    The requirement for holding all live traces ensures downstream consumers such as UI providers
    can do near-real-time partial updates.
    """

    # NOTE(nopdive): Type trickery for pydantic.
    id_node_map: Dict[int, TraceNode] = weakref.WeakValueDictionary()  # type: ignore
    node_id_map: Dict[TraceNode, int] = weakref.WeakKeyDictionary()  # type: ignore

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        weakref.finalize(self, _cleanup, f"tracehandler({id(self)})")

    def __getitem__(self, item):
        return self.id_node_map[item]

    def update_node(
        self, identifier: int, parent_id: Optional[int], node_attr: Optional[NodeAttr] = None
    ) -> TraceNode:
        """Update the trace node with the given identifier.

        If the trace node does not exist, it will be created.
        Both parent id and node attributes can be updated only once until further notice.

        Args:
            identifier: User-defined identifier for the trace node.
            parent_id: User-defined parent identifier for the trace node.
            node_attr: Input or output node attribute to be updated on the trace node.
        """
        node = self.id_node_map.get(identifier, None)
        if node is None:
            node = TraceNode()

            self.id_node_map[identifier] = node
            self.node_id_map[node] = identifier

            if parent_id is not None:
                parent_node = self.id_node_map.get(parent_id, None)
                if parent_node is not None:
                    parent_node.add_child(node)

        if node_attr is not None:
            if isinstance(node_attr, InputAttr):
                assert node.input is None
                node.input = node_attr
            elif isinstance(node_attr, OutputAttr):
                assert node.output is None
                node.output = node_attr
            else:
                raise ValueError(f"Unexpected node attr: {node_attr}")
        return node

    def root(self) -> TraceNode:
        """Returns root node of trace handler.

        Raises:
            Exception: If root cannot be found.

        Returns: Root trace node.
        """
        root = None
        for _, node in self.id_node_map.items():
            if node.parent is None:
                root = node
                break
        if root is None:
            raise Exception("No root in trace handler.")
        return root
