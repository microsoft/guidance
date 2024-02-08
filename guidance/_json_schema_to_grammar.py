import json

from typing import Dict

from ._grammar import Byte, Join, select, GrammarFunction

from .library._char_range import char_range

_QUOTE = Byte(b'"')
_SAFE_STRING = select(
    [
        char_range("a", "z"),
        char_range("A", "Z"),
        char_range("0", "9"),
        "_",
        "-",
        "'",
        " ",
    ],
    recurse=True,
)
_OPEN_BRACE = Byte(b"{")
_CLOSE_BRACE = Byte(b"}")
_COMMA = Byte(b",")
_COLON = Byte(b":")
_OPTIONAL_WHITESPACE = select([" ", ""], recurse=True)


def _process_node(node: Dict[str, any]) -> GrammarFunction:
    if node["type"] == "string":
        return Join([_QUOTE, _SAFE_STRING, _QUOTE])
    elif node["type"] == "integer":
        return Join([select(["-", ""]), select([char_range("0", "9")], recurse=True)])
    elif node["type"] == "object":
        properties = []
        for name, nxt_node in node["properties"].items():
            nxt = Join(
                [
                    Join([_QUOTE, name, _QUOTE]),
                    _OPTIONAL_WHITESPACE,
                    _COLON,
                    _OPTIONAL_WHITESPACE,
                    _process_node(nxt_node),
                ]
            )
            properties.append(nxt)
            if len(properties) < len(node["properties"]):
                properties.append(_COMMA)
                properties.append(_OPTIONAL_WHITESPACE)
        return Join([_OPEN_BRACE, *properties, _CLOSE_BRACE])
    else:
        raise ValueError(f"Unsupposed type in schema: {node['type']}")


def json_schema_to_grammar(schema: str) -> GrammarFunction:
    schema_obj = json.loads(schema)

    return _process_node(schema_obj)
