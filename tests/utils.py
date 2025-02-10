import os
import random
import time

from typing import Set, Optional, Protocol

import pytest

import guidance
from guidance import models
from guidance._grammar import GrammarFunction, Join
from guidance._parser import ByteParserException

opanai_model_cache = {}

def slowdown():
    """Limit test execution rate

    Any test calling this function will have a
    random delay inserted before the test runs.
    It can be used as a crude rate limiter for
    tests which call external APIs
    """
    delay_secs = random.randint(10, 30)
    time.sleep(delay_secs)
    return delay_secs


def remote_image_url():
    return "https://picsum.photos/300/200"


def env_or_fail(var_name: str) -> str:
    env_value = os.getenv(var_name, None)

    assert env_value is not None, f"Env '{var_name}' not found."

    return env_value

def env_or_skip(var_name: str) -> str:
    env_value = os.getenv(var_name, None)
    if env_value is None:
        pytest.skip(f"Env '{var_name}' not found.")
    return env_value


def get_model(model_name, caching=False, **kwargs):
    """Get an LLM by name."""
    if model_name.startswith("openai:"):
        return get_openai_model(model_name[7:], caching, **kwargs)
    elif model_name.startswith("transformers:"):
        return get_transformers_model(model_name[13:], caching, **kwargs)
    elif model_name.startswith("llama_cpp:"):
        return get_llama_cpp_model(model_name[10:], caching, **kwargs)
    else:
        raise ValueError(f"Could not parse '{model_name}'")


def get_openai_model(model_name, caching=False, **kwargs):
    """Get an OpenAI LLM with model reuse and smart test skipping."""

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in opanai_model_cache:
        opanai_model_cache[key] = guidance.models.OpenAI(
            model_name, caching=caching, **kwargs
        )
    lm = opanai_model_cache[key]

    return lm


transformers_model_cache = {}


def get_transformers_model(model_name, caching=False, **kwargs):
    """Get model from Hugging Face"""

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in transformers_model_cache:
        transformers_model_cache[key] = guidance.models.Transformers(
            model_name, **kwargs
        )

    return transformers_model_cache[key]


llama_cpp_model_cache = {}
llama_cpp_defaults = {"n_batch": 248}


def get_llama_cpp_model(model_name, caching=False, **kwargs):
    """Get a llama.cpp LLM with model reuse."""

    if (
        model_name is None
        or isinstance(model_name, str)
        and len(model_name.strip()) == 0
    ):
        model_name = os.environ.get("LLAMA_CPP_MODEL", "")
        if len(model_name.strip()) == 0:
            pytest.skip("No llama_cpp model found.")

    kwargs = kwargs.copy()
    for key, val in llama_cpp_defaults.items():
        if key not in kwargs:
            kwargs[key] = val

    # we cache the models so lots of tests using the same model don't have to
    # load it over and over again
    key = model_name + "_" + str(caching) + "_" + str(kwargs)
    if key not in llama_cpp_model_cache:
        llama_cpp_model_cache[key] = guidance.models.LlamaCpp(model_name, **kwargs)

    return llama_cpp_model_cache[key]


def check_match_success_with_guards(grammar, test_string: str):
    PREFIX = "A#$!"
    SUFFIX = "&%@Z"
    bracketed_grammar = Join([PREFIX, grammar, SUFFIX])

    bracketed_string = f"{PREFIX}{test_string}{SUFFIX}"

    matched = bracketed_grammar.match(bracketed_string.encode(), raise_exceptions=True)
    assert matched is not None


def check_match_failure(
    *,
    bad_string: str,
    good_bytes: Optional[bytes] = None,
    failure_byte: Optional[bytes] = None,
    allowed_bytes: Optional[Set[bytes]] = None,
    grammar: GrammarFunction,
):
    """
    Helper function to check that a string fails to match a grammar after consuming
    zero or more bytes. It checks that the consumed bytes are as expected, that the
    failure byte is as expected, and that the allowed bytes are as expected.

    allowed_bytes is allowed to be None, since it could be really complicated
    """
    with pytest.raises(ByteParserException) as pe:
        grammar.match(bad_string, raise_exceptions=True)
    if good_bytes is not None:
        assert pe.value.consumed_bytes == good_bytes
    if failure_byte is not None:
        assert pe.value.current_byte == failure_byte
    if allowed_bytes is not None:
        assert pe.value.allowed_bytes == allowed_bytes

class GrammarFunctionCallable(Protocol):
    """
    Protocol for a callable that returns a GrammarFunction and accepts
    name argument for capture key
    """

    def __call__(self, *args, name: str, **kwargs) -> GrammarFunction: ...


def generate_and_check(
    grammar_callable: GrammarFunctionCallable,
    test_string: str,
    capture_key="my_capture",
    eos_token = "<s>",
) -> models.Mock:
    # First, validate that the grammar actually accepts the test string
    grammar = grammar_callable(name=capture_key)
    match = grammar.match(test_string)
    assert match is not None
    assert match.captures[capture_key] == test_string

    # The next part is to prevent intermittent test failures
    # when the temperature is non-zero
    # The Mock model generates random characters once the
    # supplied string has been exhausted. Sometimes
    # these can also be valid according to the grammar
    # (especially when generating numbers) which interferes
    # with our round trip check.
    # So append a 'stop' character which we don't
    # use in any of our tests
    assert eos_token not in test_string, f"eos_token {eos_token!r} in string"
    prepared_string = f"{eos_token}{test_string}{eos_token}"
    lm = models.Mock(prepared_string.encode())

    # Run with the mock model
    lm += grammar

    # Make sure the round trip works
    assert lm[capture_key] == test_string

    # Return model for further checks if needed
    return lm


def check_run_with_temperature(lm: models.Model, desired_temperature: float):
    # Check on some temperatures
    if desired_temperature is not None:
        assert len(lm.engine.called_temperatures) > 0
        # Make sure that at least one temperature matches exactly
        temperature_matches = [
            x == desired_temperature for x in lm.engine.called_temperatures
        ]
        assert any(temperature_matches)
        # Check that all temperatures were 0 or the desired temperature
        # If there has been a forced byte, then get_logits() is
        # called with a temperature of zero
        assert all(
            [
                (x == desired_temperature or x == 0)
                for x in lm.engine.called_temperatures
            ]
        )
