import pathlib

from typing import Union
from urllib.parse import parse_qs, urlparse

import tiktoken

from ._grammarless import Grammarless
from ._model import Chat, Instruct
from ._openai import OpenAIEngine

try:
    import openai as openai_package

    is_openai = True
except ModuleNotFoundError:
    is_openai = False


class AzureOpenAI(Grammarless):
    """Represents an Azure OpenAI model as exposed through their remote API.

    Note that because this uses a remote API endpoint without built-in guidance support
    there are some things we cannot do, like force the model to follow a pattern inside
    a chat role block.

    Authentication can be provided via an `api_key` or through Entra/Azure Active
    Directory.
    """

    def __init__(
        self,
        model: str,
        azure_endpoint: str,
        azure_deployment: Union[str, None] = None,
        azure_ad_token_provider=None,
        api_key: Union[str, None] = None,
        tokenizer=None,
        echo: bool = True,
        version: Union[str, None] = None,
        max_streaming_tokens: int = 1000,
        timeout: float = 0.5,
        compute_log_probs: bool = False,
        **kwargs,
    ):
        """Build a new AzureOpenAI model object that represents a model in a given state.

        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-3.5-turbo).
        azure_endpoint: str
            The endpoint of the deployed model (e.g. https://my_azureai_instance.openai.azure.com)
        azure_deployment: str
            The deployed name of the model (given when the deployment was created)
        api_key: str
            The API key for calling the model
        azure_ad_token_provider:
            Alternative to the api_key, allows for use of Azure Entra authentication
        """
        if not is_openai or not hasattr(openai_package, "OpenAI"):
            raise ImportError(
                "Please install the openai package version >= 1 using `pip install openai -U` "
                "in order to use guidance.models.OpenAI!"
            )

        if api_key is None and azure_ad_token_provider is None:
            raise ValueError("Please provide either api_key or azure_ad_token_provider")

        parsed_url = urlparse(azure_endpoint)

        if azure_deployment is None:
            parts = pathlib.Path(parsed_url.path).parts
            if len(parts) > 2:
                azure_deployment = parts[3]

        parsed_query = parse_qs(parsed_url.query)
        api_version = (
            version if "api-version" not in parsed_query else parsed_query["api-version"][0]
        )

        if tokenizer is None:
            tokenizer = tiktoken.encoding_for_model(model)

        engine_instance = OpenAIEngine(
            tokenizer=tokenizer,
            max_streaming_tokens=max_streaming_tokens,
            timeout=timeout,
            compute_log_probs=compute_log_probs,
            model=model,
            azure_endpoint=f"{parsed_url.scheme}://{parsed_url.netloc}",
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            api_version=api_version,
            azure_deployment=azure_deployment,
            client_class=openai_package.AzureOpenAI,
            **kwargs,
        )

        super().__init__(
            engine_instance,
            echo=echo,
        )

    def set_logit_bias(self, logit_bias):
        self.engine.set_logit_bias(logit_bias)
