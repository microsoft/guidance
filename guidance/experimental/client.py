import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Iterable, Optional, TypeVar, Union

from openai import OpenAI
from typing_extensions import assert_never

from guidance._grammar import Function, Gen
from guidance.models import Transformers
from guidance.models._model import Engine

from .ast import ContentChunk, Node
from .state import BaseTransformersChatState, ChatState, CompletionState, State
from .state.openai import OpenAIState

S = TypeVar("S", bound=State)


class Client(ABC, Generic[S]):
    @abstractmethod
    def run(self, state: S, node: Node) -> Iterable[ContentChunk]:
        pass

    @abstractmethod
    def initial_state(self) -> S:
        pass

    def format_state(self, state: S) -> str:
        return json.dumps(state.get_state(), indent=2)


class OpenAIClient(Client[OpenAIState]):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def initial_state(self) -> OpenAIState:
        return OpenAIState.from_openai_model(self.model)

    def run(self, state: OpenAIState, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
            return

        oai_state = state.get_state()
        if oai_state["prefill"] is not None:
            raise ValueError("Prefill not supported for OpenAI")
        if oai_state["active_role"] != "assistant":
            raise ValueError("Active role must be assistant for OpenAI")

        messages = oai_state["messages"]

        if isinstance(node, Gen):
            if node.capture_name:
                raise NotImplementedError("Captures not yet supported for OpenAI")
            if node.body_regex != "(?s:.*)":
                raise ValueError("Body regex not supported for OpenAI")
            if node.stop_regex:
                raise ValueError("Stop regex not supported for OpenAI")
            if node.save_stop_text:
                raise ValueError("Save stop text not supported for OpenAI")

            responses = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=node.max_tokens,
                temperature=node.temperature,
                logprobs=True,
                stream=True,
            )
            for response in responses:
                choice = response.choices[0]
                delta = choice.delta
                if delta.content is not None:
                    yield delta.content
                    continue
                if choice.finish_reason is not None:
                    # TODO: handle finish_reason elegantly
                    break
                raise NotImplementedError(f"Unknown delta: {delta}")
        else:
            raise NotImplementedError(f"Unknown node: {node}")


class GuidanceClient(Client[S], ABC):
    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def build_prompt(self, state: S) -> str:
        pass

    def run(self, state: S, node: Node) -> Iterable[ContentChunk]:
        if isinstance(node, str):
            yield node
            return

        if isinstance(node, Function):
            prompt = self.build_prompt(state)
            engine_gen = self.engine(
                prompt,
                node,
                ensure_bos_token=False,
                echo=False,
            )
            for response in engine_gen:
                # breakpoint()
                yield response.new_bytes.decode("utf-8")
            return

        raise NotImplementedError(f"Unknown node: {node}")


class TransformersClient(GuidanceClient[Union[CompletionState, BaseTransformersChatState]]):
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct", **model_kwargs):
        self.model_id = model_id
        guidance_model = Transformers(model_id, **model_kwargs)
        super().__init__(guidance_model.engine)

    def initial_state(self) -> Union[CompletionState, BaseTransformersChatState]:
        return BaseTransformersChatState.from_model_id(self.model_id)

    def build_prompt(self, state: Union[CompletionState, BaseTransformersChatState]) -> str:
        if isinstance(state, ChatState):
            chat_state = state.get_state()
            prefill = chat_state["prefill"]
            if prefill is None:
                role = chat_state["active_role"]
                if role is None:
                    raise ValueError("Can't generate with no active role")
                prefill = {"role": "user", "content": ""}
            prompt = apply_chat_template(
                chat_state["messages"],
                chat_state["prefill"],
                None,
                None,
                self.engine.tokenizer._orig_tokenizer,
            )
        elif isinstance(state, CompletionState):
            completion_state = state.get_state()
            prompt = completion_state["prompt"]
        else:
            if TYPE_CHECKING:
                assert_never(state)
            raise TypeError(f"Expected ChatState or CompletionState, got {type(state)}")

        return prompt


from typing import Any, Optional

from .state.transformers import TransformersMessage


def apply_chat_template(
    messages: list[TransformersMessage],
    prefill: Optional[TransformersMessage],
    tools: Optional[list[Any]],
    chat_template: Optional[str],
    tokenizer: Any,
) -> str:
    if prefill is None:
        sentinel_value = None
    elif prefill["content"]:
        sentinel_value = None
        messages = messages + [prefill]
    else:
        # This is a hack to get around the fact that apply_chat_template won't properly continue the final message
        # if it is empty. We add a sentinel value to the final message, and then remove it after the fact.
        sentinel_value = "<|FINAL_MESSAGE_SENTINEL_VALUE|>"
        messages = messages + [dict(role=prefill["role"], content=sentinel_value)]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False,
    )
    if sentinel_value is not None:
        prompt = prompt[: prompt.rindex(sentinel_value)]
    return prompt
