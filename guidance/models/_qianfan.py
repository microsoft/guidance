import copy

import typing

from ._grammarless import Grammarless, GrammarlessEngine


try:
    import qianfan  # type: ignore

    client_class: typing.Optional[typing.Type[qianfan.ChatCompletion]] = qianfan.ChatCompletion
except ImportError:
    client_class = None


class ClassUnavailableException(Exception):
    pass


class QianfanAI(Grammarless):
    def __init__(
        self,
        model=None,
        echo=True,
        max_streaming_tokens=None,
        timeout=0.5,
        compute_log_probs=False,
        is_chat_model=True,
        **kwargs,
    ):
        """Build a new QianfanAI model object that represents a model in a given state."""

        if client_class is None:
            raise ClassUnavailableException("Please execute `pip install qianfan` before using QianfanAI component")

        super().__init__(
            engine=QianfanAIEngine(
                model=model,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                is_chat_model=is_chat_model,
                **kwargs,
            ),
            echo=echo,
        )


class QianfanAIEngine(GrammarlessEngine):

    def __init__(
        self,
        model,
        max_streaming_tokens,
        timeout,
        compute_log_probs,
        is_chat_model=True,
        **kwargs,
    ):
        if client_class is None:
            raise ClassUnavailableException("Please execute `pip install qianfan` before using QianfanAI component")

        assert (
            not compute_log_probs
        ), "We don't support compute_log_probs=True yet for QianfanAIEngine!"

        self.model_name = model

        self.is_chat_model = is_chat_model
        self.model_obj = qianfan.ChatCompletion(model=model, **kwargs) if self.is_chat_model else qianfan.Completion(model=model, **kwargs)

        self.extra_arguments = copy.deepcopy(kwargs)
        self.extra_arguments.pop("endpoint") if "endpoint" in kwargs else None

        super().__init__(None, max_streaming_tokens, timeout, compute_log_probs)

    def _generator(self, prompt, temperature):
        if self.is_chat_model:
            return self._chat_generator(prompt, temperature)

        return self._completion_generator(prompt, temperature)

    def _chat_generator(self, prompt, temperature):

        # find the system text
        pos = 0

        system_start = b"<|im_start|>system\n"
        user_start = b"<|im_start|>user\n"
        assistant_start = b"<|im_start|>assistant\n"
        role_end = b"<|im_end|>"

        # find the system text
        system_text = ""
        if prompt.startswith(system_start):
            pos += len(system_start)
            system_end_pos = prompt.find(role_end)
            system_text = prompt[pos:system_end_pos].decode("utf8")
            pos = system_end_pos + len(role_end)

        # find the user/assistant pairs
        messages = []
        valid_end = False
        while True:

            # find the user text
            if prompt[pos:].startswith(user_start):
                pos += len(user_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    break
                messages.append(
                    dict(
                        role="user",
                        content=prompt[pos: pos + end_pos].decode("utf8"),
                    )
                )
                pos += end_pos + len(role_end)
            elif prompt[pos:].startswith(assistant_start):
                pos += len(assistant_start)
                end_pos = prompt[pos:].find(role_end)
                if end_pos < 0:
                    valid_end = True
                    break
                messages.append(
                    dict(
                        role="assistant",
                        content=prompt[pos: pos + end_pos].decode("utf8"),
                    )
                )
                pos += end_pos + len(role_end)
            else:
                raise Exception(
                    "It looks like your prompt is not a well formed chat prompt! Please enclose all model state appends inside chat role blocks like `user()` or `assistant()`."
                )

        self._data = prompt[:pos]

        assert len(messages) > 0, "Bad chat format! No chat blocks were defined."
        assert (
                messages[-1]["role"] == "user"
        ), "Bad chat format! There must be a user() role before the last assistant() role."
        assert valid_end, "Bad chat format! You must generate inside assistant() roles."

        if temperature == 0.0:
            temperature = 0.0001

        input_kwargs = {"temperature": temperature}
        input_kwargs.update(self.extra_arguments)

        if system_text:
            input_kwargs["system"] = system_text

        input_kwargs["stream"] = True

        result_iter = self.model_obj.do(messages, **input_kwargs)
        for response in result_iter:
            yield response.body["result"].encode("utf8")

    def _completion_generator(self, prompt, temperature):
        if temperature == 0.0:
            temperature = 0.0001

        input_kwargs = {"temperature": temperature}
        input_kwargs.update(self.extra_arguments)
        input_kwargs["stream"] = True

        self._data = prompt

        result_iter = self.model_obj.do(prompt.decode("utf8"), **input_kwargs)
        for response in result_iter:
            yield response.body["result"].encode("utf8")
