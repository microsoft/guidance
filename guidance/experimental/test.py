from typing import Iterable

from .ast import ContentChunk, Node
from .client import Client, TransformersClient
from .model import Model
from .state import (
    CompletionState,
    Llama3TransformersState,
    OpenAIState,
    State,
    TransformersUnstructuredState,
)


class DummyClient(Client):
    def __init__(self, state_type: type[State]):
        self.state_type = state_type

    def initial_state(self) -> State:
        return self.state_type()

    def run(self, state: State, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
        else:
            raise NotImplementedError("Node must be a string")


def chat():
    import json

    for s in [
        OpenAIState,
        TransformersUnstructuredState,
        Llama3TransformersState,
    ]:
        model = Model(DummyClient(s))
        with model.system():
            model += "Talk like a pirate!"
        with model.user():
            model += "Hello, model!"
            model += "\nHow are you?"
        with model.assistant():
            model += "I'm doing well, thank you!"
        print("-" * 80)
        print(s.__name__)
        print("-" * 80)
        print(json.dumps(model._state.get_state(), indent=2))


def completion():
    for s in [
        CompletionState,
    ]:
        model = Model(DummyClient(s))
        model += "<|system|>\nTalk like a pirate!\n<|end_of_turn|>\n"
        model += "<|user|>\nHello, model!\n<|end_of_turn|>\n"
        model += "<|user|>\nHow are you?\n<|end_of_turn|>\n"
        model += "<|assistant|>\nI'm doing well, thank you!\n<|end_of_turn|>\n"
        print("-" * 80)
        print(s.__name__)
        print("-" * 80)
        print(model._state.get_state())


def transformers():
    from guidance import gen

    model = Model(TransformersClient())
    with model.system():
        model += "Talk like a pirate!"
    with model.user():
        model += "Hello, model!"
    with model.assistant():
        model += gen()
    return model


def vision():
    from io import BytesIO

    import PIL.Image
    import requests

    from guidance import json
    from guidance.experimental.ast import ImageBlob

    model = Model(
        TransformersClient(
            model_id="microsoft/Phi-3-vision-128k-instruct", _attn_implementation="eager"
        )
    )
    with model.system():
        model += "Talk like a pirate!"
    with model.user():
        model += ImageBlob(
            PIL.Image.open(
                BytesIO(
                    requests.get(
                        "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png"
                    ).content
                )
            )
        )
        model += "Can you describe the contents of this image?"
    with model.assistant():
        model += json(
            schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                    },
                },
                "required": ["description"],
            }
        )
    return model
