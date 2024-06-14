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


class TestLlamaCppTokenizers:
    LLAMACPP_MODELS = [
        dict(
            gguf="TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
            kwargs={"verbose": True, "n_ctx": 4096},
        ),
        dict(
            gguf="microsoft/Phi-3-mini-4k-instruct-gguf:Phi-3-mini-4k-instruct-q4.gguf",
            kwargs={"verbose": True, "n_ctx": 4096},
        ),
        dict(
            gguf="TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q8_0.gguf",
            kwargs={"verbose": True},
        ),
    ]

    def get_tokenizer(self, model_info: dict[str, Any]):
        repo_id, gguf_file = model_info["gguf"].split(":")
        downloaded_file = hf_hub_download(repo_id=repo_id, filename=gguf_file)
        lm = models.LlamaCpp(model=downloaded_file, **model_info["kwargs"])
        my_tok = lm.engine.tokenizer
        return my_tok

    @pytest.mark.parametrize("model_info", LLAMACPP_MODELS)
    def test_smoke(self, model_info: dict[str, Any]):
        my_tok = self.get_tokenizer(model_info)
        assert my_tok is not None

    @pytest.mark.parametrize("model_info", LLAMACPP_MODELS)
    @pytest.mark.parametrize("target_string", ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, model_info: dict[str, Any], target_string: str):
        my_tok = self.get_tokenizer(model_info)

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
