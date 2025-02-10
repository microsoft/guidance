import pathlib

from urllib.parse import parse_qs, urlparse

import pytest

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from guidance import assistant, gen, models, system, user

from ..model_specific import common_chat_testing
from ..utils import env_or_skip, slowdown


@pytest.fixture(scope="function")
def azureai_chat_model():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_CHAT_ENDPOINT")
    model = env_or_skip("AZUREAI_CHAT_MODEL")

    print(f"{azureai_endpoint=}")
    print(f"{model=}")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, azure_ad_token_provider=token_provider
    )
    assert isinstance(lm, models.AzureOpenAI)

    return lm


@pytest.mark.skip(reason="resource has been temporarily blocked")
def test_azureai_openai_chat_smoke(azureai_chat_model):
    common_chat_testing.smoke_chat(azureai_chat_model)


@pytest.mark.skip(reason="resource has been temporarily blocked")
def test_azureai_openai_chat_longer_1(azureai_chat_model):
    common_chat_testing.longer_chat_1(azureai_chat_model)


@pytest.mark.xfail(reason="Issue #1003")
def test_azureai_openai_chat_longer_2(azureai_chat_model):
    common_chat_testing.longer_chat_2(azureai_chat_model)


@pytest.mark.skip(reason="resource has been temporarily blocked")
def test_azureai_openai_chat_alt_args():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_CHAT_ENDPOINT")
    model = env_or_skip("AZUREAI_CHAT_MODEL")

    parsed_url = urlparse(azureai_endpoint)
    parsed_query = parse_qs(parsed_url.query)
    azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
    version = parsed_query["api-version"][0]
    min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    lm = models.AzureOpenAI(
        model=model,
        azure_endpoint=min_azureai_endpoint,
        version=version,
        azure_ad_token_provider=token_provider,
        azure_deployment=azureai_deployment,
    )

    common_chat_testing.smoke_chat(lm)


@pytest.mark.skip(reason="resource has been temporarily blocked")
def test_azureai_openai_completion_smoke():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_COMPLETION_ENDPOINT")
    model = env_or_skip("AZUREAI_COMPLETION_MODEL")

    print(f"endpoint: {' '.join(azureai_endpoint)}")
    print(f"model: {' '.join(model)}")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    lm = models.AzureOpenAI(
        model=model, azure_endpoint=azureai_endpoint, azure_ad_token_provider=token_provider
    )
    assert isinstance(lm, models.AzureOpenAI)
    assert isinstance(lm.engine, models._openai.OpenAIEngine)

    result = lm + "What is 2+2?" + gen(max_tokens=10, name="text")
    print(f"result: {result['text']}")
    assert len(result["text"]) > 0
    assert lm.engine.metrics.engine_input_tokens > 0
    assert lm.engine.metrics.engine_output_tokens > 0


@pytest.mark.skip(reason="resource has been temporarily blocked")
def test_azureai_openai_completion_alt_args():
    slowdown()

    azureai_endpoint = env_or_skip("AZUREAI_COMPLETION_ENDPOINT")
    model = env_or_skip("AZUREAI_COMPLETION_MODEL")

    parsed_url = urlparse(azureai_endpoint)
    parsed_query = parse_qs(parsed_url.query)
    azureai_deployment = pathlib.Path(parsed_url.path).parts[3]
    version = parsed_query["api-version"][0]
    min_azureai_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )

    lm = models.AzureOpenAI(
        model=model,
        azure_endpoint=min_azureai_endpoint,
        version=version,
        azure_ad_token_provider=token_provider,
        azure_deployment=azureai_deployment,
    )
    assert isinstance(lm, models.AzureOpenAI)
    assert isinstance(lm.engine, models._openai.OpenAIEngine)

    result = lm + "What is 2+2?" + gen(max_tokens=10, name="text")
    print(f"result: {result['text']}")
    assert len(result["text"]) > 0
    assert lm.engine.metrics.engine_input_tokens > 0
    assert lm.engine.metrics.engine_output_tokens > 0


@pytest.mark.skip(reason="resource has been temporarily blocked")
def test_azureai_openai_chat_loop(azureai_chat_model):
    for i in range(2):
        print(f"Iteration: {i}")
        with system():
            generation = azureai_chat_model + "You will just return whatever number I give you"

        with user():
            generation += f"The number is: {i}"

        with assistant():
            generation += gen(name="answer", max_tokens=2)

        print(str(azureai_chat_model))
        print("\n\n")
