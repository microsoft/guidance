import platform
import sys
import warnings

import numpy as np
import pytest

import guidance
from guidance import gen, select


from tests.tokenizer_common import TOKENIZER_ROUND_TRIP_STRINGS


def test_llama_cpp_gen(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm = lm + "this is a test" + gen("test", max_tokens=10)
    assert len(str(lm)) > len("this is a test")


def test_llama_cpp_gen_log_probs(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm = lm + "this is a test" + gen("test", max_tokens=1)
    assert 1 >= np.exp(lm.log_prob("test")) >= 0


def test_llama_cpp_recursion_error(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model

    # define a guidance program that adapts a proverb
    lm = (
        lm
        + f"""Tweak this proverb to apply to model instructions instead.
    {gen('verse', max_tokens=2)}
    """
    )
    assert len(str(lm)) > len("Tweak this proverb to apply to model instructions instead.\n\n")


def test_llama_cpp_select2(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm += f'this is a test1 {select(["item1", "item2"])} and test2 {select(["item3", "item4"])}'
    assert str(lm) in [
        "this is a test1 item1 and test2 item3",
        "this is a test1 item1 and test2 item4",
        "this is a test1 item2 and test2 item3",
        "this is a test1 item2 and test2 item4",
    ]


def test_repeat_calls(llamacpp_model: guidance.models.Model, selected_model_name: str):
    # This specific test is misbehaving in very specific cases, so have some
    # ugly code for conditional XFAIL
    print(f"{platform.machine()=}")
    print(f"{platform.system()=}")
    print(f"{sys.version_info=}")

    fail_combinations = [
        ("llamacpp_llama2_7b_cpu", "3.9", "Windows", "AMD64"),
        ("llamacpp_llama2_7b_cpu", "3.10", "Windows", "AMD64"),
        ("llamacpp_llama2_7b_cpu", "3.11", "Windows", "AMD64"),
        ("llamacpp_llama2_7b_cpu", "3.12", "Windows", "AMD64"),
        ("llamacpp_llama2_7b_cpu", "3.9", "Darwin", "x86_64"),
        ("llamacpp_llama2_7b_cpu", "3.10", "Darwin", "x86_64"),
        ("llamacpp_llama2_7b_cpu", "3.11", "Darwin", "x86_64"),
        ("llamacpp_llama2_7b_cpu", "3.12", "Darwin", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.9", "Windows", "AMD64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.10", "Windows", "AMD64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.11", "Windows", "AMD64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.12", "Windows", "AMD64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.9", "Darwin", "arm64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.10", "Darwin", "arm64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.11", "Darwin", "arm64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.12", "Darwin", "arm64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.9", "Darwin", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.10", "Darwin", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.11", "Darwin", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.12", "Darwin", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.9", "Linux", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.10", "Linux", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.11", "Linux", "x86_64"),
        ("llamacpp_phi3_mini_4k_instruct_cpu", "3.12", "Linux", "x86_64"),
    ]
    expect_failure = False
    python_maj_min = f"{sys.version_info[0]}.{sys.version_info[1]}"
    for f_c in fail_combinations:
        if (
            (selected_model_name == f_c[0])
            and (python_maj_min == f_c[1])
            and (platform.system() == f_c[2])
            and (platform.machine() == f_c[3])
        ):
            expect_failure = True

    try:
        llama2 = llamacpp_model
        a = []
        lm = llama2 + "How much is 2 + 2? " + gen(name="test", max_tokens=10)
        a.append(lm["test"])
        lm = llama2 + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\d+")
        a.append(lm["test"])
        lm = llama2 + "How much is 2 + 2? " + gen(name="test", max_tokens=10)
        a.append(lm["test"])
        assert a[-1] == a[0]

        assert not expect_failure, "Unexpected pass"
    except AssertionError:
        if expect_failure:
            warnings.warn("Got expected failure")
        else:
            raise


def test_suffix(llamacpp_model: guidance.models.Model):
    llama2 = llamacpp_model
    lm = llama2 + "1. Here is a sentence " + gen(name="bla", list_append=True, suffix="\n")
    assert (str(lm))[-1] == "\n"
    assert (str(lm))[-2] != "\n"


def test_subtoken_forced(llamacpp_model: guidance.models.Model):
    llama2 = llamacpp_model
    lm = llama2 + "How much is 2 + 2? " + gen(name="test", max_tokens=10, regex=r"\(")
    assert str(lm) == "How much is 2 + 2? ("


def test_llama_cpp_almost_one_batch(llamacpp_model):
    lm = llamacpp_model
    batch_size = lm.engine.model_obj.n_batch
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (batch_size - 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)


def test_llama_cpp_exactly_one_batch(llamacpp_model):
    lm = llamacpp_model
    batch_size = lm.engine.model_obj.n_batch
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * batch_size
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)


def test_llama_cpp_more_than_one_batch(llamacpp_model):
    lm = llamacpp_model
    batch_size = lm.engine.model_obj.n_batch
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (batch_size + 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)


def test_llama_cpp_almost_two_batches(llamacpp_model):
    lm = llamacpp_model
    batch_size = lm.engine.model_obj.n_batch
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * ((2 * batch_size) - 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)


def test_llama_cpp_two_batches(llamacpp_model):
    lm = llamacpp_model
    batch_size = lm.engine.model_obj.n_batch
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * (2 * batch_size)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)


def test_llama_cpp_more_than_two_batches(llamacpp_model):
    lm = llamacpp_model
    batch_size = lm.engine.model_obj.n_batch
    long_str = lm.engine.tokenizer.bos_token.decode("utf-8") * ((2 * batch_size) + 1)
    lm += long_str + gen(max_tokens=10)
    assert len(str(lm)) > len(long_str)


def test_llama_with_temp(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm += "Here is a cute 5-line poem about cats and dogs:\n"
    for i in range(5):
        lm += f"LINE {i+1}: " + gen(temperature=0.8, suffix="\n")
    # we just want to make sure we don't crash the numpy sampler


def test_llama_with_temp2(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm1 = lm + "2 + 2 =" + gen("answer", max_tokens=3)
    lm2 = lm + "2 + 2 =" + gen("answer", temperature=0.0000001, max_tokens=3)
    assert lm1["answer"] == lm2["answer"]


def test_max_tokens(llamacpp_model: guidance.models.Model):
    lm = llamacpp_model
    lm += "Who won the last Kentucky derby and by how much?"
    lm += "\n\n<<The last Kentucky Derby was held"
    lm += gen(max_tokens=2)
    assert (
        str(lm)[-1] != "<"
    )  # the output should not end with "<" because that is coming from the stop sequence...


class TestLlamaCppTokenizers:
    def test_smoke(self, llamacpp_model: guidance.models.LlamaCpp):
        my_tok = llamacpp_model.engine.tokenizer
        assert my_tok is not None

    @pytest.mark.parametrize("target_string", TOKENIZER_ROUND_TRIP_STRINGS)
    def test_string_roundtrip(self, llamacpp_model: guidance.models.LlamaCpp, target_string: str):
        my_tok = llamacpp_model.engine.tokenizer

        encoded = my_tok.encode(target_string.encode())
        decoded = my_tok.decode(encoded)
        final_string = decoded.decode()

        assert final_string == target_string
