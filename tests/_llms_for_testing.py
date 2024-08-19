import torch
import transformers

# This file contains the configurations of the models used in our test runs

# The naming convention for the keys is "<loader>_<model>_<host>" where:
# - 'loader' is 'transformers' or 'llamacpp'
# - 'model' contains relevant information about the model itself
# - 'host' is 'cpu' or 'gpu' as appropriate

_GEMMA_2 = {
    "llamacpp_gemma2_9b_cpu": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="huggingface_hubllama:bartowski/gemma-2-9b-it-GGUF:gemma-2-9b-it-IQ2_XS.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
    "transformers_gemma2_9b_cpu": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:google/gemma-2-9b-it",
        kwargs={
            "quantization_config": transformers.BitsAndBytesConfig(load_in_8bit=True),
        },
    ),
    "transformers_gemma2_9b_gpu": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:google/gemma-2-9b-it",
        kwargs={
            "device_map": "cuda:0",
            "quantization_config": transformers.BitsAndBytesConfig(load_in_4bit=True),
        },
    ),
}

_GPT_2 = {
    "transformers_gpt2_cpu": dict(name="transformers:gpt2", kwargs=dict()),
    "transformers_gpt2_gpu": dict(name="transformers:gpt2", kwargs={"device_map": "cuda:0"}),
}

_LLAMA_2 = {
    "llamacpp_llama2_7b_cpu": dict(
        name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
    "llamacpp_llama2_7b_gpu": dict(
        name="huggingface_hubllama:TheBloke/Llama-2-7B-GGUF:llama-2-7b.Q5_K_M.gguf",
        kwargs={"verbose": True, "n_gpu_layers": -1, "n_ctx": 4096},
    ),
}

_LLAMA_3 = {
    "transformers_llama3_8b_cpu": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:meta-llama/Meta-Llama-3-8B-Instruct",
        kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16},
    ),
    "transformers_llama3_8b_gpu": dict(
        # Note that this model requires an appropriate
        # HF_TOKEN environment variable
        name="transformers:meta-llama/Meta-Llama-3-8B-Instruct",
        kwargs={"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": "cuda:0"},
    ),
    "llamacpp_llama3_9b_cpu": dict(
        name="huggingface_hubllama:bartowski/Meta-Llama-3-8B-Instruct-GGUF:Meta-Llama-3-8B-Instruct-IQ3_S.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
}

_MISTRAL = {
    "transformers_mistral_7b_cpu": dict(
        name="transformers:mistralai/Mistral-7B-v0.1", kwargs=dict()
    ),
    "llamacpp_mistral_7b_cpu": dict(
        name="huggingface_hubllama:TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q8_0.gguf",
        kwargs={"verbose": True, "n_ctx": 2048},
    ),
}

_PHI_2 = {
    "transformers_phi2_cpu": dict(
        name="transformers:microsoft/phi-2", kwargs={"trust_remote_code": True}
    ),
    "transformers_phi2_gpu": dict(
        name="transformers:microsoft/phi-2",
        kwargs={"trust_remote_code": True, "device_map": "cuda:0"},
    ),
}

_PHI_3 = {
    "transformers_phi3_mini_4k_instruct_cpu": dict(
        name="transformers:microsoft/Phi-3-mini-4k-instruct",
        kwargs={"trust_remote_code": True},
    ),
    "llamacpp_phi3_mini_4k_instruct_cpu": dict(
        name="huggingface_hubllama:microsoft/Phi-3-mini-4k-instruct-gguf:Phi-3-mini-4k-instruct-q4.gguf",
        kwargs={"verbose": True, "n_ctx": 4096},
    ),
    "transformers_phi3_small_8k_instruct_gpu": dict(
        name="transformers:microsoft/Phi-3-small-8k-instruct",
        kwargs={"trust_remote_code": True, "load_in_8bit": True, "device_map": "cuda:0"},
    ),
}

AVAILABLE_MODELS = {
    "azure_guidance": dict(
        name="azure_guidance:",
        kwargs={},
    ),
}

# Add in all the other models
AVAILABLE_MODELS.update(_GEMMA_2)
AVAILABLE_MODELS.update(_GPT_2)
AVAILABLE_MODELS.update(_LLAMA_2)
AVAILABLE_MODELS.update(_LLAMA_3)
AVAILABLE_MODELS.update(_MISTRAL)
AVAILABLE_MODELS.update(_PHI_2)
AVAILABLE_MODELS.update(_PHI_3)
