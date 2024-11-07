import pytest
import json

from guidance import models, gen, select, image
from guidance._grammar import string

PHI_3_VISION_MODEL = "microsoft/Phi-3-vision-128k-instruct"

# TODO - tests with regular guidance grammars, no images

@pytest.fixture(scope="module")
def phi3_vision_model():
    """Load the TransformersPhi3Model with the specified model ID."""
    try:
        model_kwargs = {
            # "_attn_implementation": "eager", # Uncomment this line if flash attention is not working
            "trust_remote_code": True,
        }
        model = models.TransformersPhi3Vision(
            model=PHI_3_VISION_MODEL, **model_kwargs
        )
        return model
    except ImportError as e:
        pytest.skip(f"Error importing Phi 3 vision model: {e}")


def test_image_loading(phi3_vision_model: models.TransformersPhi3Vision):
    """Test basic image loading and placeholder replacement in the prompt."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "This is a test with an image: " + image(image_url)

    # Verify that the image placeholder is correctly inserted
    assert "<|image:id|>" in lm._state, f"Hidden state: {lm._state}"


def test_basic_generation_with_image(
    phi3_vision_model: models.TransformersPhi3Vision,
):
    """Test unconstrained generation with an image."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "Describe this image: " + image(image_url)
    lm += gen(name="description", max_tokens=10)

    # Verify that the model generated some text
    assert len(lm["description"]) > 0


def test_select_with_image(phi3_vision_model: models.TransformersPhi3Vision):
    """Test constraint enforcement with select and an image."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "Is this a photo of a cat or a dog: " + image(image_url)
    lm += select(["cat", "dog"], name="answer")

    # Verify that the model selected one of the options
    assert lm["answer"] in ["cat", "dog"]


def test_llguidance_interaction(phi3_vision_model: models.TransformersPhi3Vision):
    """Test that llguidance correctly enforces a simple grammar with an image."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "The color of the image is: " + image(image_url)

    # Define a grammar that only allows the words "red", "green", or "blue"
    grammar = string("red") | string("green") | string("blue")

    lm += grammar

    # Verify that the generated text is one of the allowed colors
    assert str(lm).endswith(("red", "green", "blue"))


def test_multiple_images(phi3_vision_model: models.TransformersPhi3Vision):
    """Test cache invalidation with multiple images."""
    image_url_1 = "https://picsum.photos/200/300"
    image_url_2 = "https://picsum.photos/300/200"
    lm = phi3_vision_model + "Image 1: " + image(image_url_1) + ". Image 2: " + image(image_url_2)
    lm += gen(name="description", max_tokens=10)

    # Add assertions to verify cache behavior and output (e.g., token count, presence of image tokens)
    assert lm.engine._last_image_token_position > 0
    assert len(lm["description"]) > 0


def test_empty_image_token(phi3_vision_model: models.TransformersPhi3Vision):
    """Test handling of an image token without corresponding image data."""
    with pytest.raises(KeyError) as exc_info:
        lm = (
            phi3_vision_model
            + "This is a test with a missing image: "
            + image("https://picsum.photos/200/300", id="missing_image")
        )
        lm += gen(name="description", max_tokens=10)
    # ... (Add assertions to check for expected behavior, e.g., error or default embedding generation)
    assert "Model does not contain the multimodal data with id" in str(exc_info.value)


def test_invalid_image_url(phi3_vision_model: models.TransformersPhi3Vision):
    """Test handling of an invalid image URL."""
    with pytest.raises(Exception) as exc_info:
        lm = (
            phi3_vision_model
            + "This is a test with an invalid image: "
            + image("https://invalid-image-url.com/image.jpg")
        )
        lm += gen(name="description", max_tokens=10)
    # ... (Add assertions to check for expected error handling)
    assert "Unable to load image bytes" in str(exc_info.value)


def test_complex_grammar(phi3_vision_model: models.TransformersPhi3Vision):
    """Test constraint enforcement with a more complex grammar."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "Describe this image: " + image(image_url)

    # Define a more complex grammar, potentially involving recursion or nested structures
    grammar = string("This is") | string("There is") + (string(" a ") | string(" an ")) + gen(
        name="object"
    ) + string(" in the image.")

    lm += grammar

    # Add assertions to verify that llguidance correctly enforces the grammar
    assert str(lm).startswith(("This is", "There is"))
    assert str(lm).endswith(" in the image.")


def test_token_alignment(phi3_vision_model: models.TransformersPhi3Vision):
    """Test that token alignment is maintained correctly."""
    image_url_1 = "https://picsum.photos/200/300"
    image_url_2 = "https://picsum.photos/300/200"
    lm = (
        phi3_vision_model
        + "Describe these images: "
        + image(image_url_1)
        + " and "
        + image(image_url_2)
    )

    # Force the model to generate a specific token sequence
    grammar = (
        string("Image 1 is ")
        + gen(name="color1", regex=r"[a-z]+")
        + string(". ")
        + string("Image 2 is ")
        + gen(name="color2", regex=r"[a-z]+")
        + string(".")
    )

    lm += grammar

    # Add assertions to specifically check the token sequence and verify alignment
    # For example, check that "color1" and "color2" are generated in the correct positions
    # relative to the image tokens.


def test_token_count_accuracy(phi3_vision_model: models.TransformersPhi3Vision):
    """Test the accuracy of the token count."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "Describe this image: " + image(image_url)
    lm += gen(name="description", max_tokens=10)

    # Assert that the reported token count matches the expected number of text and image tokens
    # You'll need to calculate the expected number of tokens based on the prompt, image size, and generated text.


def test_streaming_behavior(phi3_vision_model: models.TransformersPhi3Vision):
    """Test the streaming functionality with images."""
    image_url = "https://picsum.photos/200/300"
    lm = phi3_vision_model + "Describe this image: " + image(image_url)
    lm += gen(name="description", max_tokens=10)

    # Iterate over the model stream and add assertions to check partial outputs and token counts
    # For example, verify that the token count increases with each iteration and that the generated text is accumulated correctly.
