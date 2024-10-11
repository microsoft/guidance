from typing import Optional, Dict, Union
from pydantic import BaseModel

from guidance._schema import GenToken
from ..trace import NodeAttr
import json


class GuidanceMessage(BaseModel):
    class_name: str = ""

    def __init__(self, **kwargs):
        kwargs["class_name"] = self.__class__.__name__
        super().__init__(**kwargs)


class TraceMessage(GuidanceMessage):
    trace_id: int
    parent_trace_id: Optional[int]
    node_attr: Optional[NodeAttr]


class MetricMessage(GuidanceMessage):
    name: str
    value: Union[float, str, list[float], list[str]]
    scalar: bool = True


class ExecutionCompletedMessage(GuidanceMessage):
    last_trace_id: Optional[int]


class ExecutionCompletedOutputMessage(GuidanceMessage):
    trace_id: int
    text: str
    tokens: list[GenToken] = []


class ResetDisplayMessage(GuidanceMessage):
    pass


class ClientReadyMessage(GuidanceMessage):
    pass


model_registry: Dict[str, type(GuidanceMessage)] = {
    'TraceMessage': TraceMessage,
    'ExecutionCompleted': ExecutionCompletedMessage,
    'ExecutionCompletedOutputMessage': ExecutionCompletedOutputMessage,
    'ResetDisplayMessage': ResetDisplayMessage,
    'ClientReadyMessage': ClientReadyMessage,
    'MetricMessage': MetricMessage,
}


def serialize_message(message: GuidanceMessage) -> str:
    message_json = message.model_dump_json(indent=2, serialize_as_any=True)
    return message_json


def deserialize_message(data: str) -> GuidanceMessage:
    data_json = json.loads(data)
    class_name = data_json.get("class_name")
    model_class = model_registry.get(class_name)
    if not model_class:
        raise ValueError(f"Unknown class_name: {class_name}")
    return model_class.model_validate_json(data)