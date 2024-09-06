from json import dumps as json_dumps
from enum import Enum
from frozendict import frozendict, deepfreeze
from functools import cache
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Union,
    Type,
    TYPE_CHECKING,
)
import warnings

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

from .._guidance import guidance
from ..library import char_range, gen, one_or_more, optional, sequence

from .._grammar import GrammarFunction, select, capture, with_temperature
from ._pydantic import pydantic_to_json_schema
from ._subgrammar import lexeme, subgrammar


def _to_compact_json(target: Any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json_dumps(target, separators=(",", ":"))


class Keyword(str, Enum):
    ANYOF = "anyOf"
    ALLOF = "allOf"
    ONEOF = "oneOf"
    REF = "$ref"
    CONST = "const"
    ENUM = "enum"
    TYPE = "type"
    PATTERN = "pattern"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"


KEYS = {member.value for member in Keyword}

DEFS_KEYS = {"$defs", "definitions"}

IGNORED_KEYS = {
    "$schema",
    "$id",
    "id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "required",  # TODO: implement and remove from ignored list
}

# discriminator is part of OpenAPI 3.1, not JSON Schema itself
# https://json-schema.org/blog/posts/validating-openapi-and-json-schema
# TODO: While ignoring this key shouldn't lead to invalid outputs, forcing
# the model to choose the value of the marked field before other fields
# are generated (statefully or statelessly) would reduce grammar ambiguity
# and possibly improve quality.
IGNORED_KEYS.add("discriminator")

TYPE_SPECIFIC_KEYS = {
    "array": {"items", "prefixItems", "minItems", "maxItems"},
    "object": {"properties", "additionalProperties"},
}

WHITESPACE = {b" ", b"\t", b"\n", b"\r"}
STRING_CHARS = [
    char_range("a", "z"),
    char_range("A", "Z"),
    char_range("0", "9"),
    *[c for c in "-_' ,.!?/[]{}():;"],
    "\\n",
    "\\t",
    "\\\\",
]


def validate_json_node_keys(node: Mapping[str, Any]):
    keys = set(node.keys())
    valid_keys = KEYS | IGNORED_KEYS | DEFS_KEYS
    if Keyword.TYPE in node:
        valid_keys |= TYPE_SPECIFIC_KEYS.get(node[Keyword.TYPE], set())
    invalid_keys = keys - valid_keys
    if invalid_keys:
        raise ValueError(
            f"JSON schema had keys that could not be processed: {invalid_keys}" f"\nSchema: {node}"
        )


@guidance(stateless=True, cache=True)
def _gen_json_int(lm):
    return lm + lexeme(r"-?(?:0|[1-9][0-9]*)", contextual=True)


@guidance(stateless=True, cache=True)
def _gen_json_number(lm):
    return lm + select([
        _gen_json_int(),
        lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)", contextual=True),
        lexeme(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)", contextual=True),
    ])


@guidance(stateless=True, cache=True)
def _gen_json_string(
    lm,
    min_length: int = 0,
    max_length: Union[int, None] = None,
    regex: Union[str, None] = None,
):
    if regex is None:
        range_expr = f"{{{min_length},{max_length}}}" if max_length is not None else f"{{{min_length},}}"
        regex = f"(?s:.{range_expr})"
    else:
        if min_length > 0 or max_length is not None:
            msg = (
                "If a pattern is specified for a JSON "
                "string, minLength and maxLength must be "
                "left unspecified."
            )
            raise ValueError(msg)
    return lm + lexeme(regex, contextual=True, json_string=True)


@guidance(stateless=True, cache=True)
def _gen_json_object(
    lm,
    *,
    properties: frozendict[str, Any],
    additional_properties: Union[bool, frozendict[str, Any]],
    required: frozenset[str],
    definitions: frozendict[str, Callable[[], GrammarFunction]],
):
    if any(k not in properties for k in required):
        raise ValueError(f"Required properties not in properties: {required - set(properties)}")

    grammars = tuple(f'"{name}":' + _gen_json(json_schema=schema, definitions=definitions) for name, schema in properties.items())
    required_items = tuple(name in required for name in properties)

    if additional_properties is not False:
        if additional_properties is True:
            # True means that anything goes
            additional_properties = frozendict()
        additional_item_grammar =  _gen_json_string() + ':' + _gen_json(json_schema=additional_properties, definitions=definitions)
        additional_items_grammar = sequence(additional_item_grammar + ',') + additional_item_grammar
        grammars += (additional_items_grammar,)
        required_items += (False,)

    return lm + "{" + _gen_list(
        elements = grammars,
        required = required_items,
    ) + "}"

@guidance(stateless=True, cache=True)
def _gen_list(lm, *, elements: tuple[GrammarFunction, ...], required: tuple[bool, ...], prefixed: bool = False):
    if not elements:
        return lm

    elem, elements = elements[0], elements[1:]
    is_required, required = required[0], required[1:]

    if prefixed:
        if is_required:
            # If we know we have preceeding elements, we can safely just add a (',' + e)
            return lm + (',' + elem + _gen_list(elements=elements, required=required, prefixed=True))
        # If we know we have preceeding elements, we can safely just add an optional(',' + e)
        return lm + (optional(',' + elem) + _gen_list(elements=elements, required=required, prefixed=True))
    if is_required:
        # No preceding elements, and our element is required, so we just add the element
        return lm + (elem + _gen_list(elements=elements, required=required, prefixed=True))

    # No preceding elements, and our element is optional, so we add a select between the two options.
    # The first option is the recursive call with no preceding elements, the second is the recursive call
    # with the current element as a prefix.
    return lm + select([
        _gen_list(elements=elements, required=required, prefixed=False),
        elem + _gen_list(elements=elements, required=required, prefixed=True)
    ])


@guidance(stateless=True, cache=True)
def _gen_json_array(
    lm,
    *,
    prefix_items_schema: tuple[frozendict[str, Any], ...],
    item_schema: Union[bool, frozendict[str, Any]],
    min_items: int,
    max_items: Optional[int],
    definitions: frozendict[str, Callable[[], GrammarFunction]],
):
    if item_schema is True:
        # True means that anything goes
        item_schema = frozendict()

    if len(prefix_items_schema) < min_items and item_schema is False:
        raise ValueError(
            f"PrefixItems has too few elements ({len(prefix_items_schema)}) to"
            f" satisfy minItems ({min_items}) but no extra items were allowed"
        )

    if max_items is not None and max_items < min_items:
        raise ValueError(f"maxItems ({max_items}) can't be less than minItems ({min_items})")

    required_items = []
    optional_items = []

    # If max_items is None, we can add an infinite tail of items later
    n_to_add = max(len(prefix_items_schema), min_items) if max_items is None else max_items
    for i in range(n_to_add):
        if i < len(prefix_items_schema):
            schema = prefix_items_schema[i]
        elif item_schema is not False:
            schema = item_schema
        else:
            assert i >= min_items
            break

        item = _gen_json(json_schema=schema, definitions=definitions)

        if i < min_items:
            required_items.append(item)
        else:
            optional_items.append(item)

    if max_items is None and item_schema is not False:
        # Add an infinite tail of items
        item = _gen_json(json_schema=item_schema, definitions=definitions)
        optional_items.append(item + sequence("," + item))

    lm += "["

    if required_items:
        first, *rest = required_items
        lm += first
        for item in rest:
            lm += "," + item

    if optional_items:
        # This is a bit subtle and would not be required if not for prefixItems -- the previous
        # must be present before the next one may be added, meaning we have nested optionals:
        # (first optional(,second optional(,third (optional(,...)))))
        first, *rest = optional_items
        tail = ""
        for item in reversed(rest):
            tail = optional("," + item + tail)
        tail = first + tail

        if required_items:
            lm += optional("," + tail)
        else:
            lm += optional(tail)

    lm += "]"
    return lm


@guidance(stateless=True, cache=True)
def _process_anyOf(
    lm,
    *,
    anyof_list: tuple[frozendict[str, Any], ...],
    definitions: frozendict[str, Callable[[], GrammarFunction]],
):
    options = [_gen_json(json_schema=item, definitions=definitions) for item in anyof_list]
    return lm + select(options)

@guidance(stateless=True, cache=True)
def _process_allOf(
    lm,
    *,
    allof_list: tuple[frozendict[str, Any], ...],
    definitions: frozendict[str, Callable[[], GrammarFunction]],
):
    if len(allof_list) != 1:
        raise ValueError("Only support allOf with exactly one item")
    return lm + _gen_json(allof_list[0], definitions=definitions)

@guidance(stateless=True, cache=True)
def _process_oneOf(
    lm,
    *,
    oneof_list: tuple[frozendict[str, Any], ...],
    definitions: frozendict[str, Callable[[], GrammarFunction]]
):
    if len(oneof_list) == 1:
        return lm + _gen_json(oneof_list[0], definitions)
    warnings.warn("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.")
    return lm + _process_anyOf(anyof_list=oneof_list, definitions=definitions)

@guidance(stateless=True, cache=True)
def _process_const(
    lm,
    *,
    value: Any,
):
    # TODO: can we support a whitespace-flexible version of this?
    return lm + _to_compact_json(value)

@guidance(stateless=True, cache=True)
def _process_enum(lm, *, options: tuple[frozendict[str, Any], ...]):
    # TODO: can we support a whitespace-flexible version of this?
    all_opts = []
    for opt in options:
        all_opts.append(_to_compact_json(opt))
    return lm + select(options=all_opts)


@guidance(stateless=True, cache=True)
def _gen_json_any(lm):
    return lm + select(
        [
            _gen_json(json_schema=frozendict({"type": "null"}), definitions=frozendict()),
            _gen_json(json_schema=frozendict({"type": "boolean"}), definitions=frozendict()),
            _gen_json(json_schema=frozendict({"type": "integer"}), definitions=frozendict()),
            _gen_json(json_schema=frozendict({"type": "number"}), definitions=frozendict()),
            _gen_json(json_schema=frozendict({"type": "string"}), definitions=frozendict()),
            # Recursive cases
            _gen_json(
                json_schema=frozendict({
                    "type": "array",
                    "items": True,
                }),
                definitions=frozendict(),
            ),
            _gen_json(
                json_schema=frozendict({
                    "type": "object",
                    "additionalProperties": True,
                }),
                definitions=frozendict(),
            ),
        ]
    )


@guidance(stateless=True, cache=True)
def _gen_json(
    lm,
    json_schema: frozendict[str, Any],
    definitions: frozendict[str, Callable[[], GrammarFunction]],
):
    validate_json_node_keys(json_schema)

    if Keyword.ANYOF in json_schema:
        return lm + _process_anyOf(anyof_list=json_schema[Keyword.ANYOF], definitions=definitions)

    if Keyword.ALLOF in json_schema:
        return lm + _process_allOf(allof_list=json_schema[Keyword.ALLOF], definitions=definitions)

    if Keyword.ONEOF in json_schema:
        return lm + _process_oneOf(oneof_list=json_schema[Keyword.ONEOF], definitions=definitions)

    if Keyword.REF in json_schema:
        return lm + _get_definition(reference=json_schema[Keyword.REF], definitions=definitions)

    if Keyword.CONST in json_schema:
        return lm + _process_const(value=json_schema[Keyword.CONST])

    if Keyword.ENUM in json_schema:
        return lm + _process_enum(options=json_schema[Keyword.ENUM])

    if Keyword.TYPE in json_schema:
        target_type = json_schema[Keyword.TYPE]
        if target_type == "null":
            return lm + "null"
        if target_type == "boolean":
            return lm + select(["true", "false"])
        if target_type == "integer":
            return lm + _gen_json_int()
        if target_type == "number":
            return lm + _gen_json_number()
        if target_type == "string":
            return lm + _gen_json_string(
                regex=json_schema.get(Keyword.PATTERN, None),
                min_length=json_schema.get(Keyword.MIN_LENGTH, 0),
                max_length=json_schema.get(Keyword.MAX_LENGTH, None),
            )
        if target_type == "array":
            return lm + _gen_json_array(
                prefix_items_schema=json_schema.get("prefixItems", ()),
                item_schema=json_schema.get("items", True),
                min_items=json_schema.get("minItems", 0),
                max_items=json_schema.get("maxItems"),
                definitions=definitions,
            )
        if target_type == "object":
            return lm + _gen_json_object(
                properties=json_schema.get("properties", frozendict()),
                additional_properties=json_schema.get("additionalProperties", True),
                required=json_schema.get("required", frozenset()),
                definitions=definitions,
            )
        raise ValueError(f"Unsupported type in schema: {target_type}")

    return lm + _gen_json_any()


@guidance(stateless=True)
def json(
    lm,
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        Mapping[str, Any],
        Type["pydantic.BaseModel"],
        "pydantic.TypeAdapter",
    ] = None,
    compact: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 100000000,
):
    """Generate valid JSON according to the supplied JSON schema or `pydantic` model.

    Not all parts of `JSON schema <https://json-schema.org/>`_ are supported. Indeed some parts
    (such as bounds on numbers) cannot really be supported in the context of LLM generation.

    Using a JSON schema:

        >>> schema = ''{ "type": "object", "properties": { "a" : {"type": "integer"} } }'
        >>> schema_obj = json.loads(schema)
        >>> lm += json(name="generated_object", schema=schema_obj)
        >>> print(json.loads(lm["generated_object"]))
        { 'a' : 2 }

    Using a ``pydantic.BaseModel``:

        >>> class Schema(BaseModel):
        ...     b: bool
        >>> lm += json(name="generated_object", schema=Schema)
        >>> print(json.loads(lm["generated_object"]))
        { 'b' : False }

    Using a ``pydantic.TypeAdapter``:

        >>> schema = TypeAdapter(list[int])
        >>> lm += json(name="generated_object", schema=schema)
        >>> print(json.loads(lm["generated_object"]))
        [1, 2, 3]

    Parameters
    ----------

    name : str or None
        If this is not None then the the results of the generation will be saved as a variable on
        the Model object (so you can access the result as ``lm["var_name"]``).

    schema : Union[None, Mapping[str, Any], Type[pydantic.BaseModel], pydantic.TypeAdapter]
        One of:
            - None, in which case any valid JSON will be generated
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``

    compact : bool
        If True, the generated JSON will be forced to be compact (no whitespace).
        If False, output will be whitespace-flexible (i.e. decided by the model).
    """
    if isinstance(schema, Mapping):
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        jsonschema.validators.Draft202012Validator.check_schema(schema)
    elif schema is None:
        schema = {}
    else:
        schema = pydantic_to_json_schema(schema)

    # Freeze the schema to make it immutable and hashable
    frozen_schema: frozendict = deepfreeze(schema)

    definitions: frozendict[str, Callable[[], GrammarFunction]] = frozendict()
    for dk in DEFS_KEYS:
        if dk in frozen_schema:
            assert len(definitions) == 0, "Found duplicate definitions"
            definitions = _build_definitions(frozen_schema[dk])

    return lm + with_temperature(
        subgrammar(
            name,
            body=_gen_json(json_schema=frozen_schema, definitions=definitions),
            skip_regex=(
                None if compact
                else r"[\x20\x0A\x0D\x09]+"
            ),
            no_initial_skip=True,
            max_tokens=max_tokens,
        ),
        temperature=temperature,
    )

@cache
def _build_definitions(
    raw_definitions: frozendict[str, Any]
) -> frozendict[str, Callable[[], GrammarFunction]]:
    definitions: frozendict[str, Callable[[], GrammarFunction]]

    def build_definition(json_schema: frozendict[str, Any]) -> Callable[[], GrammarFunction]:
        @guidance(stateless=True, dedent=False, cache=True)
        def closure(lm):
            return lm + _gen_json(json_schema=json_schema, definitions=definitions)

        return closure

    definitions = frozendict({ref: build_definition(schema) for ref, schema in raw_definitions.items()})
    return definitions


def _get_definition(
    reference: str,
    definitions: frozendict[str, Callable[[], GrammarFunction]],
) -> GrammarFunction:
    assert definitions is not None
    target_definition = None
    for dk in DEFS_KEYS:
        ref_start = f"#/{dk}/"
        if reference.startswith(ref_start):
            target_name = reference[len(ref_start) :]
            target_definition = definitions[target_name]

    assert target_definition is not None
    return target_definition()
