import pytest

from guidance import models


from ..model_specific import common_chat_testing
from ..utils import env_or_fail, slowdown

pytest.skip("Deployments temporarily deleted", allow_module_level=True)

# How to fill out the environment variables to
# set up the models
# Temporarily remove mistral pending endpoint investigation
_chat_models = {"phi3": "PHI3", "llama3": "LLAMA3_CHAT"}


def _get_chat_model(model_name: str):
    env_string = _chat_models[model_name]

    azureai_studio_endpoint = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_ENDPOINT")
    azureai_studio_deployment = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_DEPLOYMENT")
    azureai_studio_key = env_or_fail(f"AZURE_AI_STUDIO_{env_string}_KEY")

    lm = models.AzureAIStudioChat(
        azureai_studio_endpoint=azureai_studio_endpoint,
        azureai_studio_deployment=azureai_studio_deployment,
        azureai_studio_key=azureai_studio_key,
        clear_cache=True,
    )
    assert isinstance(lm, models.AzureAIStudioChat)
    return lm


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_smoke(chat_model_name: str):
    slowdown()

    lm = _get_chat_model(chat_model_name)

    common_chat_testing.smoke_chat(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_1(chat_model_name: str):
    slowdown()

    lm = _get_chat_model(chat_model_name)
    common_chat_testing.longer_chat_1(lm, chat_model_name != "mistral")


@pytest.mark.parametrize("chat_model_name", _chat_models.keys())
def test_azureai_chat_longer_2(chat_model_name: str):
    slowdown()

    lm = _get_chat_model(chat_model_name)
    common_chat_testing.longer_chat_2(lm, chat_model_name != "mistral")
