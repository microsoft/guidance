import pytest
from unittest.mock import patch

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


def test_token_count(selected_model):
    lm = selected_model
    lm2 = lm + " 1 1 1 1 1" + gen(max_tokens=9) + gen(max_tokens=9)
    assert (
        18 <= lm2.token_count <= 20
    )  # note we allow ourselves to be off by one because it is hard to know when we are continuing vs starting a new token in the parser


def test_token_healing(selected_model):
    """Tests a bug where the space is incorrectly forced as token 220, while it should be not forced it might be extended"""
    model_obj = getattr(selected_model.engine, "model_obj", None)
    if model_obj is None or type(model_obj).__name__ != "GPT2LMHeadModel":
        pytest.skip("Test for GPT2 bug only")
    gpt2 = selected_model
    lm = gpt2 + (
        "This is a story of 10 or 5 or " + zero_or_more(byte_range(b"0", b"9"))
    )
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


def test_associativity(selected_model):
    prompt = "pi = "
    grammar = gen("number", regex=r"\d")
    engine = selected_model.engine

    with patch.object(engine, "get_next_token", side_effect=engine.get_next_token) as get_next_token_1:
        _ = selected_model + (prompt + grammar)
    prompt_tokens_1 = get_next_token_1.call_args.kwargs["token_ids"]

    with patch.object(engine, "get_next_token", side_effect=engine.get_next_token) as get_next_token_2:
        _ = (selected_model + prompt) + grammar
    prompt_tokens_2 = get_next_token_2.call_args.kwargs["token_ids"]

    # Main assertion: the prompt tokens should be the same
    assert prompt_tokens_1 == prompt_tokens_2

    # Further assert that the tokenization matches the expected tokenization
    expected_prompt_tokens = engine.tokenizer.encode(prompt.encode())
    if (
        engine.tokenizer.bos_token is not None
        and expected_prompt_tokens[:1] != [engine.tokenizer.bos_token_id]
    ):
        expected_prompt_tokens = [engine.tokenizer.bos_token_id] + expected_prompt_tokens
        expected_prompt_tokens = engine.tokenizer.recode(expected_prompt_tokens)
    assert prompt_tokens_1 == expected_prompt_tokens
