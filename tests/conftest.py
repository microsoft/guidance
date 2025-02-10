import os
import pytest

from guidance import models

def pytest_addoption(parser):
    SELECTED_MODEL_ENV_VARIABLE = "GUIDANCE_SELECTED_MODEL"
    default_model = os.getenv(SELECTED_MODEL_ENV_VARIABLE, "transformers_gpt2_cpu")
    parser.addoption(
        "--selected_model",
        action="store",
        default=default_model,
        type=str,
        help=f"LLM to load when needed. Set default via environment variable {SELECTED_MODEL_ENV_VARIABLE}",
    )


@pytest.fixture(scope="session")
def selected_model_name(pytestconfig) -> str:
    return pytestconfig.getoption("selected_model")


@pytest.fixture(scope="session")
def selected_model(selected_model_name: str) -> models.Model:
    """Get a concrete model for tests

    This fixture is for tests which are supposed
    to run against any LLM supported by guidance.
    Rather than loading a model themselves, each test
    can just use this fixture, and get a 'live'
    model.

    When running the tests, the model used is
    controlled by the '--selected_model' command
    line argument to pytest.

    The naming convention for the keys is "<loader>_<model>_<host>" where:
    - 'loader' is 'transformers' or 'llamacpp'
    - 'model' contains relevant information about the model itself
    - 'host' is 'cpu' or 'gpu' as appropriate
    """

    # GEMMA 2
    if selected_model_name == "llamacpp_gemma2_9b_cpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="bartowski/gemma-2-9b-it-GGUF", filename="gemma-2-9b-it-IQ2_XS.gguf"
            ),
            verbose=True,
            n_ctx=4096,
        )
    if selected_model_name == "transformers_gemma2_9b_cpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from transformers import BitsAndBytesConfig

        return models.Transformers(
            "google/gemma-2-9b-it",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
    if selected_model_name == "transformers_gemma2_9b_gpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from transformers import BitsAndBytesConfig

        return models.Transformers(
            "google/gemma-2-9b-it",
            device_map="cuda:0",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

    # GPT 2
    if selected_model_name == "transformers_gpt2_cpu":
        return models.Transformers("gpt2")
    if selected_model_name == "transformers_gpt2_gpu":
        return models.Transformers("gpt2", device_map="cuda:0")

    # LLAMA 2
    if selected_model_name == "llamacpp_llama2_7b_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q5_K_M.gguf"),
            verbose=True,
            n_ctx=4096,
        )
    if selected_model_name == "llamacpp_llama2_7b_gpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(repo_id="TheBloke/Llama-2-7B-GGUF", filename="llama-2-7b.Q5_K_M.gguf"),
            verbose=True,
            n_ctx=4096,
            n_gpu_layers=-1,
        )

    # LLAMA 3
    if selected_model_name == "transformers_llama3_8b_cpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from torch import bfloat16

        return models.Transformers(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=bfloat16,
        )
    if selected_model_name == "transformers_llama3_8b_gpu":
        # Note that this model requires an appropriate HF_TOKEN environment variable
        from torch import bfloat16

        return models.Transformers(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=bfloat16,
            device_map="cuda:0",
        )

    # MISTRAL
    if selected_model_name == "transformers_mistral_7b_cpu":
        return models.Transformers("mistralai/Mistral-7B-v0.1")
    if selected_model_name == "llamacpp_mistral_7b_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                filename="mistral-7b-instruct-v0.2.Q8_0.gguf",
            ),
            verbose=True,
            n_ctx=2048,
        )

    # PHI 2
    if selected_model_name == "transformers_phi2_cpu":
        return models.Transformers("microsoft/phi-2", trust_remote_code=True)
    if selected_model_name == "transformers_phi2_gpu":
        return models.Transformers("microsoft/phi-2", trust_remote_code=True, device_map="cuda:0")

    # PHI 3
    if selected_model_name == "transformers_phi3_mini_4k_instruct_cpu":
        return models.Transformers("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
    if selected_model_name == "llamacpp_phi3_mini_4k_instruct_cpu":
        from huggingface_hub import hf_hub_download

        return models.LlamaCpp(
            hf_hub_download(
                repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
                filename="Phi-3-mini-4k-instruct-q4.gguf",
            ),
            verbose=True,
            n_ctx=4096,
        )
    if selected_model_name == "transformers_phi3_small_8k_instruct_gpu":
        return models.Transformers(
            "microsoft/Phi-3-small-8k-instruct",
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="cuda:0",
        )

    # QWEN2DOT5
    if selected_model_name == "transformers_qwen2dot5_0dot5b_cpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B")
    if selected_model_name == "transformers_qwen2dot5_0dot5b_gpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B", device_map="cuda:0")
    if selected_model_name == "transformers_qwen2dot5_0dot5b_instruct_cpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B-Instruct")
    if selected_model_name == "transformers_qwen2dot5_0dot5b_instruct_gpu":
        return models.Transformers("Qwen/Qwen2.5-0.5B-Instruct", device_map="cuda:0")

    raise ValueError(
        f"No support for selected_model_name {selected_model_name}"
    )  # pragma: no cover


@pytest.fixture(scope="module")
def llamacpp_model(selected_model, selected_model_name):
    if selected_model_name in [
        "llamacpp_llama2_7b_cpu",
        "llamacpp_llama2_7b_gpu",
        "llamacpp_gemma2_9b_cpu",
        "llamacpp_phi3_mini_4k_instruct_cpu",
        "llamacpp_mistral_7b_cpu",
    ]:
        return selected_model
    else:
        pytest.skip("Requires Llama-Cpp model")
