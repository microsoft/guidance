import pytest

import guidance
from guidance import byte_range, gen, models, select, zero_or_more


def test_fstring(selected_model):
    lm = selected_model
    lm += f'this is a test {select(["item1", "item2"])}'
    assert str(lm) in ["this is a test item1", "this is a test item2"]


def test_fstring_custom(selected_model):
    lm = selected_model

    @guidance
    def my_function(lm):
        return lm + f'another {select(["item1", "item2"])}'

    lm += f"this is a test {my_function()}"
    assert str(lm) in ["this is a test another item1", "this is a test another item2"]


def test_with_multitokenchars(selected_model: guidance.models.Model):
    # Taken from https://github.com/guidance-ai/guidance/issues/934
    lm = selected_model
    lm += "歪" + select(["打正着", "门邪道"])
    assert str(lm) == "歪打正着" or str(lm) == "歪门邪道"


def test_token_count(selected_model):
    lm = selected_model
    lm2 = lm + " 1 1 1 1 1" + gen(max_tokens=9) + gen(max_tokens=9)
    assert (
        18 <= lm2.token_count <= 20
    )  # note we allow ourselves to be off by one because it is hard to know when we are continuing vs starting a new token in the parser


def test_token_healing(selected_model):
    """Tests a bug where the space is incorrectly forced as token 220, while it should be not forced it might be extended"""
    model_type = type(selected_model.engine.model_obj).__name__
    print(model_type)
    if model_type != "GPT2LMHeadModel":
        pytest.skip("Test for GPT2 bug only")
    gpt2 = selected_model
    lm = gpt2 + ("This is a story of 10 or 5 or " + zero_or_more(byte_range(b"0", b"9")))
    assert len(lm) > len("This is a story of 10 or 5 or ")


def test_stream(selected_model):
    lm = selected_model.stream()
    lm += select(["item1", "item2"])
    *_, last_lm = lm
    assert str(last_lm) in ["item1", "item2"]


def test_stream_propagate_errors(selected_model):
    lm = selected_model.stream()

    @guidance
    def my_function(lm):
        raise Exception()

    lm += my_function()
    with pytest.raises(Exception):
        list(lm)


def test_stream_add_multiple(selected_model):
    """Test to make sure multiple additions to a ModelStream are all respected"""
    lm = selected_model.stream()
    lm += select(["item1", "item2"])
    lm += ""
    *_, last_lm = lm
    assert str(last_lm) in ["item1", "item2"]
