from json import dumps as json_dumps, loads as json_loads
from enum import Enum
import math
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Union,
    Type,
    TYPE_CHECKING,
    cast,
)
import warnings
import referencing
from collections import defaultdict
import urllib.parse

def urijoin(base: str, uri: str) -> str:
    # Special case for fragment-only URIs
    if uri.startswith("#"):
        return f"{base}{uri}"
    return urllib.parse.urljoin(base, uri)

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

from .._guidance import guidance
from ..library import char_range, gen, one_or_more, optional, sequence
from ..library._regex_utils import rx_int_range, rx_float_range

from .._grammar import GrammarFunction, select, capture, with_temperature, Not, And, quote_regex
from ._pydantic import pydantic_to_json_schema
from ._subgrammar import as_regular_grammar, lexeme, subgrammar

JSONValue = Union[None, bool, int, float, str, Mapping[str, "JSONValue"], Sequence["JSONValue"]]
JSONSchema = Union[bool, Mapping[str, JSONValue]]

class Unset(Enum):
    # https://peps.python.org/pep-0484/#support-for-singleton-types-in-unions
    token = 0
_unset = Unset.token

DRAFT202012_RESERVED_KEYWORDS = {
    # Anchors and References
    '$anchor',
    '$dynamicAnchor',
    '$dynamicRef',
    '$id',
    '$recursiveAnchor',
    '$recursiveRef',
    '$ref',
    '$schema',
    '$vocabulary',

    # Schema Structure and Combining Schemas
    '$defs',
    'allOf',
    'anyOf',
    'definitions',
    'dependencies',
    'dependentRequired',
    'dependentSchemas',
    'else',
    'if',
    'not',
    'oneOf',
    'then',

    # Validation Keywords for Any Instance Type
    'const',
    'enum',
    'type',

    # Validation Keywords for Numeric Instances
    'exclusiveMaximum',
    'exclusiveMinimum',
    'maximum',
    'minimum',
    'multipleOf',

    # Validation Keywords for Strings
    'format',
    'maxLength',
    'minLength',
    'pattern',

    # Validation Keywords for Arrays
    'contains',
    'items',
    'maxContains',
    'maxItems',
    'minContains',
    'minItems',
    'prefixItems',
    'uniqueItems',

    # Validation Keywords for Objects
    'additionalProperties',
    'maxProperties',
    'minProperties',
    'patternProperties',
    'properties',
    'propertyNames',
    'required',
    'unevaluatedItems',
    'unevaluatedProperties',

    # Metadata Keywords
    '$comment',
    'default',
    'deprecated',
    'description',
    'examples',
    'readOnly',
    'title',
    'writeOnly',

    # Content Validation
    'contentEncoding',
    'contentMediaType',
    'contentSchema',
}

class JSONType(str, Enum):
    NULL = "null"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"

class Keyword(str, Enum):
    ANYOF = "anyOf"
    ALLOF = "allOf" # Note: Partial support. Only supports exactly one item.
    ONEOF = "oneOf" # Note: Partial support. This is converted to anyOf.
    ID = "$id"
    REF = "$ref"
    CONST = "const"
    ENUM = "enum"
    TYPE = "type"

class NumberKeywords(str, Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXCLUSIVE_MINIMUM = "exclusiveMinimum"
    EXCLUSIVE_MAXIMUM = "exclusiveMaximum"

class StringKeywords(str, Enum):
    PATTERN = "pattern"
    FORMAT = "format"
    MIN_LENGTH = "minLength"
    MAX_LENGTH = "maxLength"

class ArrayKeywords(str, Enum):
    PREFIX_ITEMS = "prefixItems"
    ITEMS = "items"
    MIN_ITEMS = "minItems"
    MAX_ITEMS = "maxItems"

class ObjectKeywords(str, Enum):
    PROPERTIES = "properties"
    ADDITIONAL_PROPERTIES = "additionalProperties"
    REQUIRED = "required"

TYPE_SPECIFIC_KEYWORDS = {
    JSONType.INTEGER: NumberKeywords,
    JSONType.NUMBER: NumberKeywords,
    JSONType.STRING: StringKeywords,
    JSONType.ARRAY: ArrayKeywords,
    JSONType.OBJECT: ObjectKeywords,
}

IGNORED_KEYS = {
    "$anchor",
    "$defs",
    "$schema",
    "id",
    "$comment",
    "title",
    "default",
    "definitions",
    "description",
    "examples",
}

# discriminator is part of OpenAPI 3.1, not JSON Schema itself
# https://json-schema.org/blog/posts/validating-openapi-and-json-schema
# TODO: While ignoring this key shouldn't lead to invalid outputs, forcing
# the model to choose the value of the marked field before other fields
# are generated (statefully or statelessly) would reduce grammar ambiguity
# and possibly improve quality.
IGNORED_KEYS.add("discriminator")

WHITESPACE = {b" ", b"\t", b"\n", b"\r"}
VALID_KEYS = set(Keyword) | set(NumberKeywords) | set(StringKeywords) | set(ArrayKeywords) | set(ObjectKeywords) | IGNORED_KEYS

FORMAT_PATTERNS: dict[str, Optional[str]] = {
    # https://json-schema.org/understanding-json-schema/reference/string#built-in-formats
    # Dates and times
    "date-time": (
        r'(?P<date>[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01]))'
        r'[tT]'
        r'(?P<time>'
            r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)'
            r'(?P<time_fraction>\.[0-9]+)?'
            r'(?P<time_zone>[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])'
        r')'
    ),
    "time": (
        r'(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)'
        r'(?P<time_fraction>\.[0-9]+)?'
        r'(?P<time_zone>[zZ]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])'
    ),
    "date": r'[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])',
    "duration": (
        r'P'                                     # Start with 'P'
        r'(?:'                                   # Non-capturing group for main alternatives
            r'(?P<dur_date>'                     # Named group for date duration
                r'(?:'                           # Non-capturing group for date components
                    r'(?P<dur_year>'             # Named group for years
                        r'[0-9]+Y'                  # One or more digits followed by 'Y'
                        r'(?:'                   # Optional month
                            r'[0-9]+M'              # One or more digits followed by 'M'
                            r'(?:[0-9]+D)?'         # Optional days
                        r')?'
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_month>'            # Named group for months
                        r'[0-9]+M'                  # One or more digits followed by 'M'
                        r'(?:[0-9]+D)?'             # Optional days
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_day>'              # Named group for days
                        r'[0-9]+D'                  # One or more digits followed by 'D'
                    r')'
                r')'
                r'(?:'                           # Optional time
                    r'T'                         # Time starts with 'T'
                    r'(?:'                       # Non-capturing group for time components
                        r'(?P<dur_hour>'         # Named group for hours
                            r'[0-9]+H'              # One or more digits followed by 'H'
                            r'(?:'               # Optional minutes
                                r'[0-9]+M'          # One or more digits followed by 'M'
                                r'(?:[0-9]+S)?'     # Optional seconds
                            r')?'
                        r')'
                        r'|'                     # OR
                        r'(?P<dur_minute>'       # Named group for minutes
                            r'[0-9]+M'              # One or more digits followed by 'M'
                            r'(?:[0-9]+S)?'         # Optional seconds
                        r')'
                        r'|'                     # OR
                        r'(?P<dur_second>'       # Named group for seconds
                            r'[0-9]+S'              # One or more digits followed by 'S'
                        r')'
                    r')'
                r')?'
            r')'
            r'|'                                 # OR
            r'(?P<dur_time>'                     # Named group for time-only duration
                r'T'                             # Time starts with 'T'
                r'(?:'                           # Non-capturing group for time components
                    r'(?P<dur_hour2>'             # Named group for hours
                        r'[0-9]+H'                  # One or more digits followed by 'H'
                        r'(?:'                   # Optional minutes
                            r'[0-9]+M'              # One or more digits followed by 'M'
                            r'(?:[0-9]+S)?'         # Optional seconds
                        r')?'
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_minute2>'           # Named group for minutes
                        r'[0-9]+M'                  # One or more digits followed by 'M'
                        r'(?:[0-9]+S)?'             # Optional seconds
                    r')'
                    r'|'                         # OR
                    r'(?P<dur_second2>'           # Named group for seconds
                        r'[0-9]+S'                  # One or more digits followed by 'S'
                    r')'
                r')'
            r')'
            r'|'                                 # OR
            r'(?P<dur_week>'                     # Named group for weeks
                r'[0-9]+W'                          # One or more digits followed by 'W'
            r')'
        r')'
    ),
    # Email addresses
    "email": (
        r'(?P<local_part>'
            r'(?P<dot_string>'
                r'[^\s@\.]+'
                r'(\.[^\s@\.]+)*'
            r')'
            # TODO: Add support for quoted strings
        r')'
        r'@'
        r'('
            r'(?P<domain>'
                r'(?P<sub_domain>'
                    r'[a-zA-Z0-9]'
                    r'([a-zA-Z0-9-]*[a-zA-Z0-9])?'
                r')'
                r'(\.(?P<sub_domain2>'
                    r'[a-zA-Z0-9]'
                    r'([a-zA-Z0-9-]*[a-zA-Z0-9])?'
                r'))*'
            r')'
            r'|' # OR
            r'\[(?P<ipv4>((([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))\.){3}(([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9]))))\]'
        r')'
    ),
    "idn-email": None,
    # Hostnames
    "hostname": r"[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*",
    "idn-hostname": None,
    "ipv4": r'((([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))\.){3}(([0-9])|(([1-9])[0-9]|(25[0-5]|(2[0-4]|(1)[0-9])[0-9])))',
    "ipv6": (
        # Full IPv6 address without "::"
        r'(?:'
            r'(?P<full>(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        # Leading "::" (shortens leading zeros)
        r'(?:'
            r'::(?:[0-9a-fA-F]{1,4}:){0,5}(?P<ls32>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        # "::" within the address, and variants reducing the length of the address
        r'(?:'
            r'(?P<h16_1>[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,4}(?P<ls32_1>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,1}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,3}(?P<ls32_2>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,2}[0-9a-fA-F]{1,4})?::(?:[0-9a-fA-F]{1,4}:){0,2}(?P<ls32_3>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,3}[0-9a-fA-F]{1,4})?::[0-9a-fA-F]{1,4}:(?P<ls32_4>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,4}[0-9a-fA-F]{1,4})?::(?P<ls32_5>[0-9a-fA-F]{1,4}:[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4})?::(?P<h16_2>[0-9a-fA-F]{1,4})'
        r')'
        r'|'  # OR
        r'(?:'
            r'((?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4})?::'
        r')'
    ),
    # Resource identifiers
    "uuid": (
        r'(?P<time_low>[0-9a-fA-F]{8})'      # 4 hex octets for time-low
        r'-'                                 # Literal hyphen
        r'(?P<time_mid>[0-9a-fA-F]{4})'      # 2 hex octets for time-mid
        r'-'                                 # Literal hyphen
        r'(?P<time_high_and_version>[0-9a-fA-F]{4})'  # 2 hex octets for time-high-and-version
        r'-'                                 # Literal hyphen
        r'(?P<clock_seq_and_reserved>[0-9a-fA-F]{2})' # 1 hex octet for clock-seq-and-reserved
        r'(?P<clock_seq_low>[0-9a-fA-F]{2})' # 1 hex octet for clock-seq-low
        r'-'                                 # Literal hyphen
        r'(?P<node>[0-9a-fA-F]{12})'         # 6 hex octets for node
    ),
    "uri": None,
    "uri-reference": None,
    "iri": None,
    "iri-reference": None,
    # URI template
    "uri-template": None,
    # JSON pointers
    "json-pointer": None,
    "relative-json-pointer": None,
    # Regular expressions
    "regex": None, # Might need a full CFG?,
    # Unknown
    "unknown": r"(?s:.*)",
}

def _get_format_pattern(format: str) -> str:
    try:
        pattern = FORMAT_PATTERNS[format]
    except KeyError:
        raise ValueError(f"Format {format!r} is not supported")
    if pattern is None:
        raise NotImplementedError(f"Format {format!r} is not yet supported")
    return pattern


def validate_json_node_keys(node: Mapping[str, Any]):
    keys = set(node.keys())
    # Any key that is a valid JSON schema keyword but not one that we have explicit support for is "invalid"
    invalid_keys = (keys - VALID_KEYS).intersection(DRAFT202012_RESERVED_KEYWORDS)
    if invalid_keys:
        raise ValueError(
            f"JSON schema had keys that could not be processed: {invalid_keys}" f"\nSchema: {node}"
        )


def get_sibling_keys(node: Mapping[str, Any], key: str) -> set[str]:
    # Get the set of functional (non-ignored) keys that are siblings of the given key
    return set(node.keys()) & VALID_KEYS - set(IGNORED_KEYS) - {key}


class GenJson:
    item_separator = ", "
    key_separator = ": "
    def __init__(self, schema: JSONSchema, separators: Optional[tuple[str, str]] = None) -> None:
        self.schema = schema
        if separators is not None:
            self.item_separator, self.key_separator = separators

        registry: referencing.Registry[JSONSchema] = referencing.Registry()
        resource: referencing.Resource[JSONSchema] = referencing.jsonschema.DRAFT202012.create_resource(schema)
        self._base_uri = resource.id() or ""
        registry = registry.with_resource(
            uri=self._base_uri,
            resource=resource
        )
        self._resolver = registry.resolver()
        self._defs: dict[str, Callable[[], GrammarFunction]] = {}


    @guidance(stateless=True)
    def ref(
        self,
        lm,
        *,
        reference: str,
        base_uri: str,
    ):
        """
        Resolve a reference to another schema and return the grammar for that schema.

        Note: we define a zero-argument closure that will return the grammar for the reference and
        add it to the _defs cache. This allows us to avoid re-resolving the reference every time
        and to handle recursive references correctly.
        """
        abspath = urijoin(base_uri, reference)

        if abspath not in self._defs:
            resolved = self._resolver.lookup(abspath)
            base_uri_of_resolved = resolved.resolver._base_uri

            @guidance(stateless=True, dedent=False, cache=True)
            def closure(lm):
                grammar = self.json(json_schema=resolved.contents, base_uri=base_uri_of_resolved)
                return lm + grammar

            self._defs[abspath] = closure
        return lm + self._defs[abspath]()


    @guidance(stateless=True)
    def root(self, lm):
        return lm + self.json(json_schema=self.schema, base_uri=self._base_uri)


    @classmethod
    @guidance(stateless=True)
    def integer(cls, lm, minimum: Union[float, int, None] = None, maximum: Union[float, int, None] = None, exclusiveMinimum: bool = False, exclusiveMaximum: bool = False):
        if minimum is not None:
            if exclusiveMinimum:
                if minimum != int(minimum):
                    minimum = math.ceil(minimum)
                else:
                    minimum += 1
            else:
                minimum = math.ceil(minimum)
            minimum = int(minimum)
        if maximum is not None:
            if exclusiveMaximum:
                if maximum != int(maximum):
                    maximum = math.floor(maximum)
                else:
                    maximum -= 1
            else:
                maximum = math.floor(maximum)
            maximum = int(maximum)

        return lm + lexeme(rx_int_range(minimum, maximum), contextual=True)


    @classmethod
    @guidance(stateless=True)
    def number(cls, lm, minimum: Optional[float] = None, maximum: Optional[float] = None, exclusiveMinimum: bool = False, exclusiveMaximum: bool = False):
        return lm + lexeme(
            rx_float_range(
                minimum, maximum,
                left_inclusive = not exclusiveMinimum,
                right_inclusive = not exclusiveMaximum
            ),
            contextual=True
        )


    @classmethod
    @guidance(stateless=True)
    def string(
        cls,
        lm,
        *,
        min_length: int = 0,
        max_length: Union[int, None] = None,
        regex: Union[str, None] = None,
        format: Union[str, None] = None,
    ):
        if (regex is not None or format is not None) and (min_length > 0 or max_length is not None):
            raise ValueError(
                "If a pattern or format is specified for a JSON string, minLength and maxLength must be left unspecified."
            )

        if regex is not None and format is not None:
            raise ValueError("Cannot specify both a regex and a format for a JSON string")

        if format is not None:
            regex = _get_format_pattern(format)

        elif regex is not None:
            # Sanitize the regex, removing unnecessary anchors that may cause problems later
            # NOTE/TODO: this could potentially be pushed further down into the lexeme function,
            # but it's not immediately clear whether anchors in other contexts are superfluous.
            regex = regex.lstrip("^").rstrip("$")

        elif regex is None:
            range_expr = f"{{{min_length},{max_length}}}" if max_length is not None else f"{{{min_length},}}"
            regex = f"(?s:.{range_expr})"

        return lm + lexeme(regex, contextual=True, json_string=True)


    @guidance(stateless=True)
    def object(
        self,
        lm,
        *,
        properties: Mapping[str, JSONSchema],
        additional_properties: JSONSchema,
        required: Sequence[str],
        base_uri: str,
    ):
        # "required" keys will be validated against "properties" if they're present, otherwise against "additionalProperties".
        # If "additionalProperties" is False, then required keys must be in "properties".
        if any(k not in properties for k in required) and additional_properties is False:
            raise ValueError(
                f"Required properties not in properties but additionalProperties is False."
                f" Missing required properties: {list(r for r in required if r not in properties)}"
            )

        keys: list[str] = []
        required_items: list[bool] = []
        grammars: list[GrammarFunction] = []
        # First iterate over the properties in order, then iterate over any missing required keys, using additional_properties as the schema
        for name in (*properties, *(r for r in required if r not in properties)):
            # Use json_dumps to properly quote / escape the key
            key = json_dumps(name)
            keys.append(key)
            # Identify if the key is required
            required_items.append(name in required)
            # Build the grammar we'll use for this property
            grammars.append(f'{key}{self.key_separator}' + self.json(json_schema=properties.get(name, additional_properties), base_uri=base_uri))

        if additional_properties is not False:
            # Key for additionalProperties is a json string, but we need to disallow any properties that are already defined
            additional_key_grammar: GrammarFunction
            if len(keys) > 0:
                additional_key_grammar = as_regular_grammar(
                    And([
                        lexeme(r'"([^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'),
                        Not(lexeme('|'.join(map(quote_regex, keys)))),
                    ]),
                    lexeme = True,
                )
            else:
                additional_key_grammar = self.string()

            additional_item_grammar = additional_key_grammar + self.key_separator + self.json(json_schema=additional_properties, base_uri=base_uri)
            additional_items_grammar = sequence(additional_item_grammar + self.item_separator) + additional_item_grammar
            grammars.append(additional_items_grammar)
            required_items.append(False)

        return lm + "{" + self._join(
            elements = tuple(grammars),
            required = tuple(required_items),
        ) + "}"


    @guidance(stateless=True, cache=True)
    def _join(self, lm, *, elements: tuple[GrammarFunction, ...], required: tuple[bool, ...], prefixed: bool = False):
        if not elements:
            return lm

        elem, elements = elements[0], elements[1:]
        is_required, required = required[0], required[1:]

        if prefixed:
            if is_required:
                # If we know we have preceeding elements, we can safely just add a (',' + e)
                return lm + (self.item_separator + elem + self._join(elements=elements, required=required, prefixed=True))
            # If we know we have preceeding elements, we can safely just add an optional(',' + e)
            return lm + (optional(self.item_separator + elem) + self._join(elements=elements, required=required, prefixed=True))
        if is_required:
            # No preceding elements, and our element is required, so we just add the element
            return lm + (elem + self._join(elements=elements, required=required, prefixed=True))

        # No preceding elements, and our element is optional, so we add a select between the two options.
        # The first option is the recursive call with no preceding elements, the second is the recursive call
        # with the current element as a prefix.
        return lm + select([
            self._join(elements=elements, required=required, prefixed=False),
            elem + self._join(elements=elements, required=required, prefixed=True)
        ])


    @guidance(stateless=True)
    def array(
        self,
        lm,
        *,
        prefix_items_schema: Sequence[JSONSchema],
        item_schema: JSONSchema,
        min_items: int,
        max_items: Optional[int],
        base_uri: str,
    ):
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

            item = self.json(json_schema=schema, base_uri=base_uri)

            if i < min_items:
                required_items.append(item)
            else:
                optional_items.append(item)

        if max_items is None and item_schema is not False:
            # Add an infinite tail of items
            item = self.json(json_schema=item_schema, base_uri=base_uri)
            optional_items.append(item + sequence(self.item_separator + item))

        lm += "["

        if required_items:
            first, *rest = required_items
            lm += first
            for item in rest:
                lm += self.item_separator + item

        if optional_items:
            # This is a bit subtle and would not be required if not for prefixItems -- the previous
            # must be present before the next one may be added, meaning we have nested optionals:
            # (first optional(,second optional(,third (optional(,...)))))
            first, *rest = optional_items
            tail: Union[str, GrammarFunction] = ""
            for item in reversed(rest):
                tail = optional(self.item_separator + item + tail)
            tail = first + tail

            if required_items:
                lm += optional(self.item_separator + tail)
            else:
                lm += optional(tail)

        lm += "]"
        return lm


    @guidance(stateless=True)
    def anyOf(
        self,
        lm,
        *,
        anyof_list: Sequence[JSONSchema],
        base_uri: str,
    ):
        options = [self.json(json_schema=item, base_uri=base_uri) for item in anyof_list]
        return lm + select(options)

    @guidance(stateless=True)
    def oneOf(
        self,
        lm,
        *,
        oneof_list: Sequence[JSONSchema],
        base_uri: str,
    ):
        if len(oneof_list) == 1:
            return lm + self.json(json_schema=oneof_list[0], base_uri=base_uri)
        warnings.warn("oneOf not fully supported, falling back to anyOf. This may cause validation errors in some cases.")
        return lm + self.anyOf(anyof_list=oneof_list, base_uri=base_uri)

    @guidance(stateless=True)
    def allOf(
        self,
        lm,
        *,
        parent_schema: JSONSchema,
        base_uri: str,
    ):
        type = set(JSONType)
        properties: defaultdict[str, list[JSONSchema]] = defaultdict(list)
        required: set[str] = set()
        additional_properties_list: list[JSONSchema] = []
        items_list: list[JSONSchema] = []
        other_data: dict[str, JSONValue] = {}
        enum: Optional[list[JSONValue]] = None
        const: Union[Unset, JSONValue] = _unset

        def handle_keyword(key: str, value: JSONValue, base_uri: str):
            nonlocal type
            nonlocal required
            nonlocal const
            nonlocal enum

            if key == Keyword.REF:
                ref = cast(str, value)
                abspath = urijoin(base_uri, ref)
                resolved = self._resolver.lookup(abspath)
                add_schema(resolved.contents, base_uri=resolved.resolver._base_uri)

            elif key == Keyword.CONST:
                value = cast(JSONValue, value)
                if const is not _unset and const != value:
                    raise ValueError(f"allOf with multiple conflicting const values: {const!r} and {value!r}")
                const = value

            elif key == Keyword.ENUM:
                value = cast(Sequence[JSONValue], value)
                if enum is not None:
                    try:
                        enum = list(set(enum) & set(value))
                    except TypeError:
                        # Check on equality, not on hash
                        # Yes, this is O(n^2).
                        # Hope the items were unique.
                        # ¯\_(ツ)_/¯
                        enum = [a for a in enum if a == b for b in value]
                else:
                    enum = value

            elif key == Keyword.TYPE:
                value = cast(Union[str, Sequence[str]], value)
                if isinstance(value, str):
                    value_set = {value}
                else:
                    value_set = set(value)
                if JSONType.NUMBER in value_set:
                    # Number implies integer
                    value_set.add(JSONType.INTEGER)
                type &= value_set
                # Throw an error early if we have conflicting types
                if not type:
                    raise ValueError("allOf with conflicting types")

            elif key == Keyword.ALLOF:
                value = cast(Sequence[JSONSchema], value)
                for schema in value:
                    add_schema(schema, base_uri)

            elif key == ObjectKeywords.PROPERTIES:
                value = cast(Mapping[str, JSONSchema], value)
                for name, schema in value.items():
                    this_base_uri = schema.get(Keyword.ID, base_uri)
                    if Keyword.REF in schema:
                        # Make the ref absolute so that it can be resolved in the right scope later
                        schema = schema.copy()
                        schema[Keyword.REF] = urijoin(this_base_uri, schema[Keyword.REF])
                    properties[name].append(schema)

            elif key == ObjectKeywords.REQUIRED:
                value = cast(Sequence[str], value)
                required |= set(value)

            elif key == ObjectKeywords.ADDITIONAL_PROPERTIES:
                # TODO: do the additionalProperties of one schema need to evaluate against the properties of another?
                # TODO: unevaluatedProperties?
                value = cast(JSONSchema, value)
                additional_properties_list.append(value)

            elif key == ArrayKeywords.ITEMS:
                value = cast(JSONSchema, value)
                items_list.append(value)

            elif key in set(Keyword):
                # If we've done our job right, we should never hit this case...
                raise NotImplementedError(f"Don't yet know how to handle {key} in allOf")

            elif key in other_data:
                raise NotImplementedError(f"Don't yet know how to reduce multiple values of {key!r} in allOf")

            else:
                other_data[key] = value

        def add_schema(schema: JSONSchema, base_uri: str):
            if schema is True:
                return
            if schema is False:
                raise ValueError("allOf contains a False schema")

            # Validate the schema's keys (we have only validated the parent schema's keys so far)
            # TODO: This will make us validate the parent twice... should probably be refactored
            validate_json_node_keys(schema)

            # Set the base_uri for this schema
            if Keyword.ID in schema:
                # TODO: avoid copies if possible..?
                schema = schema.copy()
                base_uri = urijoin(base_uri, schema.pop(Keyword.ID))

            for key, value in schema.items():
                if key in IGNORED_KEYS:
                    continue
                handle_keyword(key, value, base_uri)

        add_schema(parent_schema, base_uri)

        combined_schema = {
            Keyword.TYPE: list(type),
        }
        if properties:
            combined_schema[ObjectKeywords.PROPERTIES] = {}
            for name, schemas in properties.items():
                if len(schemas) == 1:
                    combined_schema[ObjectKeywords.PROPERTIES][name] = schemas[0]
                else:
                    combined_schema[ObjectKeywords.PROPERTIES][name] = {"allOf": schemas}
        if required:
            combined_schema[ObjectKeywords.REQUIRED] = required
        if additional_properties_list:
            if len(additional_properties_list) == 1:
                combined_schema[ObjectKeywords.ADDITIONAL_PROPERTIES] = additional_properties_list[0]
            else:
                combined_schema[ObjectKeywords.ADDITIONAL_PROPERTIES] = {"allOf": additional_properties_list}
        if items_list:
            if len(items_list) == 1:
                combined_schema[ArrayKeywords.ITEMS] = items_list[0]
            else:
                combined_schema[ArrayKeywords.ITEMS] = {"allOf": items_list}
        if enum is not None:
            combined_schema[Keyword.ENUM] = enum
        if const is not _unset:
            combined_schema[Keyword.CONST] = const

        assert not set(combined_schema) & set(other_data)
        combined_schema.update(other_data)

        return lm + self.json(json_schema=combined_schema, base_uri=base_uri)


    @guidance(stateless=True)
    def const(
        self,
        lm,
        *,
        value: Union[None, bool, int, float, str, Mapping, Sequence],
        instance_type: Optional[Union[str, Sequence[str]]] = None,
        enum: Optional[Sequence[Union[None, bool, int, float, str, Mapping, Sequence]]] = None,
    ):
        schema_to_validate_against: dict[str, Any] = {}
        if instance_type is not None:
            schema_to_validate_against["type"] = instance_type
        if enum is not None:
            schema_to_validate_against["enum"] = enum
        if schema_to_validate_against:
            # Raise a validation error if the value doesn't match the type
            jsonschema.validate(
                instance=value,
                schema=schema_to_validate_against,
            )
        # Base case
        if isinstance(value, (type(None), bool, int, float, str)):
            return lm + json_dumps(value)
        # Recursive cases
        # NOTE: we could potentially just use json_dumps in all cases, but this will ensure that we're
        # properly treating all parts as individual lexemes, which makes whitespace flexibility possible
        if isinstance(value, Mapping):
            return lm + self.json(
                json_schema={
                    "type": "object",
                    "properties": {k: {"const": v} for k, v in dict(value).items()},
                    "required": list(value.keys()),
                    "additionalProperties": False,
                },
                base_uri="", # dummy value -- we don't need to resolve anything
            )
        if isinstance(value, Sequence):
            return lm + self.json(
                json_schema={
                    "type": "array",
                    "prefixItems": [{"const": v} for v in list(value)],
                    "minItems": len(value),
                    "maxItems": len(value),
                    "items": False,
                },
                base_uri="", # dummy value -- we don't need to resolve anything
            )
        raise TypeError(f"Unsupported value type: {type(value)} for value: {value!r}")

    @guidance(stateless=True)
    def enum(
        self,
        lm,
        *,
        options: Sequence[Union[None, bool, int, float, str, Mapping, Sequence]],
        instance_type: Optional[Union[str, Sequence[str]]] = None,
    ):
        all_opts: list[GrammarFunction] = []
        for instance in options:
            try:
                grm = self.const(value=instance, instance_type=instance_type)
            except jsonschema.ValidationError:
                continue
            all_opts.append(grm)
        if not all_opts:
            raise ValueError(f"No valid options found for enum with type {instance_type!r}: {options}")
        return lm + select(options=all_opts)


    @guidance(stateless=True)
    def any(self, lm):
        return lm + select(
            [
                # Dummy base uris ok since we're not resolving anything
                self.json(json_schema={"type": "null"}, base_uri=""),
                self.json(json_schema={"type": "boolean"}, base_uri=""),
                self.json(json_schema={"type": "integer"}, base_uri=""),
                self.json(json_schema={"type": "number"}, base_uri=""),
                self.json(json_schema={"type": "string"}, base_uri=""),
                # Recursive cases
                self.json(
                    json_schema={
                        "type": "array",
                        "items": True,
                    },
                    base_uri="",
                ),
                self.json(
                    json_schema={
                        "type": "object",
                        "additionalProperties": True,
                    },
                    base_uri="",
                ),
            ]
        )


    @guidance(stateless=True)
    def json(
        self,
        lm,
        *,
        json_schema: JSONSchema,
        base_uri: str,
    ):
        if json_schema is True:
            json_schema = {}
        elif json_schema is False:
            raise ValueError("No valid JSON can be generated from a schema of `False`")

        if json_schema == {}:
            return lm + self.any()

        validate_json_node_keys(json_schema)

        if Keyword.ID in json_schema:
            # "cd" into the new base_uri
            base_uri = urijoin(base_uri, json_schema[Keyword.ID])

        if Keyword.ALLOF in json_schema and Keyword.ANYOF in json_schema:
            parent_schema = json_schema.copy()
            anyof_list = parent_schema.pop(Keyword.ANYOF)
            allof_list = parent_schema.pop(Keyword.ALLOF)
            # Reduce the problem to an anyOf of allOfs
            return lm + self.anyOf(
                anyof_list=[
                    {"allOf": [any_item, *allof_list], **parent_schema}
                    for any_item in anyof_list
                ],
                base_uri=base_uri,
            )

        if Keyword.ALLOF in json_schema and Keyword.ONEOF in json_schema:
            parent_schema = json_schema.copy()
            allof_list = parent_schema.pop(Keyword.ALLOF)
            oneof_list = parent_schema.pop(Keyword.ONEOF)
            # Reduce the problem to a oneOf of allOfs
            return lm + self.oneOf(
                oneof_list=[
                    {"allOf": [one_item, *allof_list], **parent_schema}
                    for one_item in oneof_list
                ],
                base_uri=base_uri,
            )

        if Keyword.ANYOF in json_schema and Keyword.ONEOF in json_schema:
            parent_schema = json_schema.copy()
            anyof_list = parent_schema.pop(Keyword.ANYOF)
            oneof_list = parent_schema.pop(Keyword.ONEOF)
            assert Keyword.ALLOF not in parent_schema
            # Reduce the problem to a oneOf of allOfs
            return lm + self.oneOf(
                oneof_list=[
                    {"allOf": [one_item, any_item], **parent_schema}
                    for any_item in anyof_list
                    for one_item in oneof_list
                ],
                base_uri=base_uri,
            )

        if Keyword.ALLOF in json_schema:
            return lm + self.allOf(parent_schema=json_schema, base_uri=base_uri)

        if Keyword.ANYOF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.ANYOF)
            if not sibling_keys:
                return lm + self.anyOf(anyof_list=json_schema[Keyword.ANYOF], base_uri=base_uri)
            # Let the allOf function handle anyOfs with sibling keys
            parent_schema = json_schema.copy()
            anyof_list = parent_schema.pop(Keyword.ANYOF)
            return lm + self.anyOf(
                anyof_list=[
                    {"allOf": [any_item], **parent_schema}
                    for any_item in anyof_list
                ],
                base_uri=base_uri,
            )

        if Keyword.ONEOF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.ONEOF)
            if not sibling_keys:
                return lm + self.oneOf(oneof_list=json_schema[Keyword.ONEOF], base_uri=base_uri)
            # Let the allOf function handle oneOfs with sibling keys
            parent_schema = json_schema.copy()
            oneof_list = parent_schema.pop(Keyword.ONEOF)
            assert Keyword.ALLOF not in parent_schema
            return lm + self.oneOf(
                oneof_list=[
                    {"allOf": [one_item], **parent_schema}
                    for one_item in oneof_list
                ],
                base_uri=base_uri,
            )

        if Keyword.REF in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.REF)
            if not sibling_keys:
                return lm + self.ref(reference=json_schema[Keyword.REF], base_uri=base_uri)
            # Let the allOf function handle refs with sibling keys
            parent_schema = json_schema.copy()
            ref = parent_schema.pop(Keyword.REF)
            assert Keyword.ALLOF not in parent_schema
            return lm + self.allOf(parent_schema={"allOf": [{Keyword.REF: ref}], **parent_schema}, base_uri=base_uri)

        if Keyword.CONST in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.CONST) - {Keyword.TYPE, Keyword.ENUM}
            if sibling_keys:
                raise NotImplementedError(f"const with sibling keys is not yet supported. Got {sibling_keys}")
            return lm + self.const(value=json_schema[Keyword.CONST], instance_type=json_schema.get(Keyword.TYPE, None), enum=json_schema.get(Keyword.ENUM, None))

        if Keyword.ENUM in json_schema:
            sibling_keys = get_sibling_keys(json_schema, Keyword.ENUM) - {Keyword.TYPE}
            if sibling_keys:
                raise NotImplementedError(f"enum with sibling keys is not yet supported. Got {sibling_keys}")
            return lm + self.enum(options=json_schema[Keyword.ENUM], instance_type=json_schema.get(Keyword.TYPE, None))

        if Keyword.TYPE in json_schema:
            target_types = cast(Union[str, Sequence[str]], json_schema[Keyword.TYPE])
            if isinstance(target_types, str):
                target_types = [target_types]
        else:
            target_types = list(JSONType)

        options: list[Union[str, GrammarFunction]] = []
        option: Union[str, GrammarFunction]
        for target_type in target_types:
            if target_type == JSONType.NULL:
                option = "null"
            elif target_type == JSONType.BOOLEAN:
                option = select(["true", "false"])
            elif target_type in {JSONType.INTEGER, JSONType.NUMBER}:
                minimum = cast(Union[int, float, None], json_schema.get(NumberKeywords.MINIMUM, None))
                maximum = cast(Union[int, float, None], json_schema.get(NumberKeywords.MAXIMUM, None))
                # Older schemas (Draft4) may have exclusiveMinimum and exclusiveMaximum as booleans, but Draft202012+ should have them as numbers
                exclusive_minimum = cast(Union[int, float, None], json_schema.get(NumberKeywords.EXCLUSIVE_MINIMUM, None))
                exclusive_maximum = cast(Union[int, float, None], json_schema.get(NumberKeywords.EXCLUSIVE_MAXIMUM, None))
                # Internally, we'll use Draft4 style booleans
                exclusive_minimum_flag: bool = False
                exclusive_maximum_flag: bool = False

                if exclusive_minimum is not None:
                    if minimum is None or exclusive_minimum >= minimum:
                        minimum = exclusive_minimum
                        exclusive_minimum_flag = True

                if exclusive_maximum is not None:
                    if maximum is None or exclusive_maximum <= maximum:
                        maximum = exclusive_maximum
                        exclusive_maximum_flag = True

                if target_type == JSONType.INTEGER:
                    option = self.integer(
                        minimum=minimum,
                        maximum=maximum,
                        exclusiveMinimum=exclusive_minimum_flag,
                        exclusiveMaximum=exclusive_maximum_flag,
                    )
                else:
                    option = self.number(
                        minimum=minimum,
                        maximum=maximum,
                        exclusiveMinimum=exclusive_minimum_flag,
                        exclusiveMaximum=exclusive_maximum_flag,
                    )
            elif target_type == JSONType.STRING:
                option = self.string(
                    regex=json_schema.get(StringKeywords.PATTERN, None),
                    format=json_schema.get(StringKeywords.FORMAT, None),
                    min_length=json_schema.get(StringKeywords.MIN_LENGTH, 0),
                    max_length=json_schema.get(StringKeywords.MAX_LENGTH, None),
                )
            elif target_type == JSONType.ARRAY:
                option = self.array(
                    prefix_items_schema=json_schema.get(ArrayKeywords.PREFIX_ITEMS, []),
                    item_schema=json_schema.get(ArrayKeywords.ITEMS, True),
                    min_items=json_schema.get(ArrayKeywords.MIN_ITEMS, 0),
                    max_items=json_schema.get(ArrayKeywords.MAX_ITEMS, None),
                    base_uri=base_uri,
                )
            elif target_type == JSONType.OBJECT:
                option = self.object(
                    properties=json_schema.get(ObjectKeywords.PROPERTIES, {}),
                    additional_properties=json_schema.get(ObjectKeywords.ADDITIONAL_PROPERTIES, True),
                    required=json_schema.get(ObjectKeywords.REQUIRED, set()),
                    base_uri=base_uri,
                )
            else:
                raise ValueError(f"Unsupported type in schema: {target_type}")
            options.append(option)

        return lm + select(options)


@guidance(stateless=True)
def json(
    lm,
    name: Optional[str] = None,
    *,
    schema: Union[
        None,
        str,
        JSONSchema,
        Type["pydantic.BaseModel"],
        "pydantic.TypeAdapter",
    ] = None,
    temperature: float = 0.0,
    max_tokens: int = 100000000,
    separators: Optional[tuple[str, str]] = None,
    whitespace_flexible: bool = False,
    **kwargs,
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
            - A string representing a JSON schema which will be parsed using ``json.loads()``
            - A JSON schema object. This is a JSON schema string which has been passed to ``json.loads()``
            - A subclass of ``pydantic.BaseModel``
            - An instance of ``pydantic.TypeAdapter``
    """
    if "compact" in kwargs:
        warnings.warn("The 'compact' argument is deprecated and has no effect. It will be removed in a future release.", category=DeprecationWarning)
        kwargs.pop("compact")
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {kwargs.keys()}")
    if schema is None:
        # Default schema is empty, "anything goes" schema
        # TODO: consider default being `{"type": "object"}`
        schema = {}
    elif isinstance(schema, (Mapping, bool, str)):
        if isinstance(schema, str):
            schema = cast(JSONSchema, json_loads(schema))
        # Raises jsonschema.exceptions.SchemaError or ValueError
        # if schema is not valid
        jsonschema.validators.Draft202012Validator.check_schema(schema)
    elif isinstance(schema, pydantic.TypeAdapter) or (isinstance(schema, type) and issubclass(schema, pydantic.BaseModel)):
        schema = pydantic_to_json_schema(schema)
    else:
        raise TypeError(f"Unsupported schema type: {type(schema)}")

    if whitespace_flexible:
        if separators is None:
            separators = (",", ":")
        skip_regex = r"[\x20\x0A\x0D\x09]+"
    else:
        skip_regex = None

    return lm + with_temperature(
        subgrammar(
            name,
            body=GenJson(schema=schema, separators=separators).root(),
            skip_regex=skip_regex,
            no_initial_skip=True,
            max_tokens=max_tokens,
        ),
        temperature=temperature,
    )
