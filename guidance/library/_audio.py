import pathlib
import typing
import pkg_resources
import base64

from .._guidance import guidance
from .._utils import bytes_from
from ..trace._trace import AudioOutput
# from ..trace._trace import AudioInput


@guidance
def audio(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    bytes_data = bytes_from(src, allow_local=allow_local)
    base64_string = base64.b64encode(bytes_data).decode('utf-8')
    lm += AudioOutput(value=base64_string, is_input=True)
    # lm += AudioInput(value=base64_string)
    return lm


@guidance
def gen_audio(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    with open(
        pkg_resources.resource_filename("guidance", "resources/sample_audio.wav"), "rb"
    ) as f:
        bytes_data = f.read()
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        lm += AudioOutput(value=base64_string, is_input=False)
    return lm