import pytest
import uuid
import requests
import tempfile
import pathlib

from urllib.error import HTTPError, URLError
from guidance import models, image
from ...utils import remote_image_url


def test_local_image():
    model = models.Mock()
    with tempfile.TemporaryDirectory() as temp_dir:
        td = pathlib.Path(temp_dir)
        filename = f"{str(uuid.uuid4())}.jpg"
        fullname = td / filename
        with open(fullname, "wb") as file:
            response = requests.get(remote_image_url())
            file.write(response.content)
        assert (fullname).exists()
        model += image(fullname)
        assert str(model).startswith("<|_image:")


def test_local_image_not_found():
    model = models.Mock()
    with pytest.raises(FileNotFoundError):
        model += image("not_found.jpg")


def test_remote_image():
    model = models.Mock()
    model += image(remote_image_url())

    assert str(model).startswith("<|_image:")


def test_remote_image_not_found():
    model = models.Mock()
    with pytest.raises((HTTPError, URLError)):
        model += image("https://example.com/not_found.jpg")


def test_image_from_bytes():
    model = models.Mock()
    with tempfile.TemporaryDirectory() as temp_dir:
        td = pathlib.Path(temp_dir)
        filename = f"{str(uuid.uuid4())}.jpg"
        fullname = td / filename
        with open(fullname, "wb") as file:
            response = requests.get(remote_image_url())
            file.write(response.content)
        assert (fullname).exists()
        with open(fullname, "rb") as f:
            model += image(f.read())
            assert str(model).startswith("<|_image:")
