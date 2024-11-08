from json import loads as json_loads, dumps as json_dumps
from typing import (
    Any,
    Mapping,
    Optional,
    Union,
    Type,
    TYPE_CHECKING,
    cast,
)
import warnings

from llguidance import JsonCompiler

from .._grammar import LLGrammar, with_temperature, capture
from ._pydantic import pydantic_to_json_schema

try:
    import jsonschema
    import pydantic
except ImportError:
    if TYPE_CHECKING:
        raise

JSONSchema = Mapping[str, Any]

def json(
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

    compiler = JsonCompiler(
        separators=separators,
        whitespace_flexible=whitespace_flexible,
    )
    
    g = LLGrammar(
        json_loads(compiler.compile(json_dumps(schema))),
        max_tokens=max_tokens,
    )
    if name is not None:
        g = capture(g, name)
    return with_temperature(g, temperature)
