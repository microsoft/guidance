import os
import pathlib
import random
import tempfile
import time
import uuid

import pytest
import requests

from guidance import models

# Ensure that asserts from tests/utils.py are rewritten by pytest to show helpful messages
pytest.register_assert_rewrite("tests.utils")

from ._llms_for_testing import AVAILABLE_MODELS
from .utils import get_model

SELECTED_MODEL_ENV_VARIABLE = "GUIDANCE_SELECTED_MODEL"


def pytest_addoption(parser):
    default_model = os.getenv(SELECTED_MODEL_ENV_VARIABLE, "transformers_gpt2_cpu")
    parser.addoption(
        "--selected_model",
        action="store",
        default=default_model,
        type=str,
        choices=AVAILABLE_MODELS.keys(),
        help=f"LLM to load when needed. Set default via environment variable {SELECTED_MODEL_ENV_VARIABLE}",
    )


@pytest.fixture(scope="session")
def selected_model_name(pytestconfig) -> str:
    return pytestconfig.getoption("selected_model")


@pytest.fixture(scope="session")
def selected_model_info(selected_model_name: str):
    model_info = AVAILABLE_MODELS[selected_model_name]
    return model_info


@pytest.fixture(scope="module")
def selected_model(selected_model_info: str) -> models.Model:
    """Get a concrete model for tests

    This fixture is for tests which are supposed
    to run against any LLM supported by guidance.
    Rather than loading a model themselves, each test
    can just use this fixture, and get a 'live'
    model.

    When running the tests, the model used is
    controlled by the '--selected_model' command
    line argument to pytest.
    """
    model = get_model(selected_model_info["name"], **(selected_model_info["kwargs"]))
    assert model is not None
    return model


@pytest.fixture(scope="module")
def llamacpp_model(selected_model, selected_model_name):
    if selected_model_name in [
        "llamacpp_llama2_7b_cpu",
        "llamacpp_llama2_7b_gpu",
        "llamacpp_llama3_9b_cpu",
        "llamacpp_gemma2_9b_cpu",
        "llamacpp_phi3_mini_4k_instruct_cpu",
        "llamacpp_mistral_7b_cpu",
    ]:
        return selected_model
    else:
        pytest.skip("Requires Llama-Cpp model")


@pytest.fixture(scope="function")
def rate_limiter() -> int:
    """Limit test execution rate

    Any test using this fixture will have a
    random delay inserted before the test runs.
    It can be used as a crude rate limiter for
    tests which call external APIs
    """
    delay_secs = random.randint(10, 30)
    time.sleep(delay_secs)
    return delay_secs


@pytest.fixture(scope="session")
def remote_image_url():
    return "https://picsum.photos/300/200"


@pytest.fixture(scope="session")
def local_image_path(remote_image_url):
    with tempfile.TemporaryDirectory() as temp_dir:
        td = pathlib.Path(temp_dir)
        filename = f"{str(uuid.uuid4())}.jpg"
        with open(td / filename, "wb") as file:
            response = requests.get(remote_image_url)
            file.write(response.content)
        assert (td / filename).exists()
        yield td / filename


@pytest.fixture(scope="session")
def local_image_bytes(local_image_path):
    with open(local_image_path, "rb") as f:
        return f.read()
