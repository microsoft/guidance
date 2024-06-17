import pytest

from typing import Any

import tiktoken

from huggingface_hub import hf_hub_download

from guidance import models


ROUND_TRIP_STRINGS = [
    "",
    " ",
    "hello",
    " hello",
    "two words",
    " two words",
    " two words ",
    "two words ",
    "’",
]


class TestTransformerTokenizers:
    TRANSFORMER_MODELS = [
        "gpt2",
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-vision-128k-instruct",
        "microsoft/phi-2",
    ]

    @pytest.mark.parametrize(
        "model_name",
        TRANSFORMER_MODELS,
    )
    def test_smoke(self, model_name: str):
        my_tok = models.TransformersTokenizer(model=model_name, transformers_tokenizer=None)
        assert my_tok is not None

    @pytest.mark.parametrize("model_name", TRANSFORMER_MODELS)
    @pytest.mark.parametrize("target_string", ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_name: str, target_string: str):
        my_tok = models.TransformersTokenizer(model=model_name, transformers_tokenizer=None)

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string


class TestTiktoken:
    MODELS = ["gpt-3.5-turbo", "gpt-4"]

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("target_string", ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_name: str, target_string: str):
        my_tik = tiktoken.encoding_for_model(model_name)
        my_tok = models._grammarless.GrammarlessTokenizer(my_tik)

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string
