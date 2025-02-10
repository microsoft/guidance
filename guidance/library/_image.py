import pathlib
import typing
import pkg_resources
import base64

from .._guidance import guidance
from .._utils import bytes_from
from ..trace._trace import ImageOutput
# from ..trace._trace import ImageInput


@guidance
def image(lm, src: typing.Union[str, pathlib.Path, bytes], allow_local: bool = True):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    bytes_data = bytes_from(src, allow_local=allow_local)
    base64_string = base64.b64encode(bytes_data).decode('utf-8')
    lm += ImageOutput(value=base64_string, is_input=True)
    # lm += ImageInput(value=base64_string)
    return lm

    # NOTE(nopdive): Older code for image integration commented out.
    # bytes_data = bytes_from(src, allow_local=allow_local)
    # bytes_id = str(id(bytes_data))
    #
    # # set the image bytes
    # lm = lm.set(bytes_id, bytes_data)
    # lm += f"<|_image:{bytes_id}|>"
    # return lm


@guidance
def gen_image(lm):
    # TODO(nopdive): Mock for testing. Remove all of this code later.
    with open(
            pkg_resources.resource_filename("guidance", "resources/sample_image.png"), "rb"
    ) as f:
        bytes_data = f.read()
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        lm += ImageOutput(value=base64_string, is_input=False)
    return lm
