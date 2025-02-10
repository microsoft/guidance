import typing

import tiktoken

from ._grammarless import Grammarless, GrammarlessEngine

try:
    import openai

    client_class: typing.Optional[typing.Type[openai.OpenAI]] = openai.OpenAI
except ImportError:
    client_class = None


class OpenAIEngine(GrammarlessEngine):
    def __init__(
        self,
        tokenizer,
        max_streaming_tokens,
        timeout,
        compute_log_probs,
        model,
        client_class=client_class,
        **kwargs,
    ):
        self.logit_bias = None

        if client_class is None:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )

        self.client = client_class(**kwargs)
        self.model_name = model

        # Create a simple registry of models that use completion endpoints.
        self._completion_models = set(
            [
                "gpt-35-turbo-instruct",
                "gpt-3.5-turbo-instruct",
                "babbage-002",
                "davinci-002",
            ]
        )

        if tokenizer is None:
            tokenizer = tiktoken.encoding_for_model(model)

        super().__init__(tokenizer, max_streaming_tokens, timeout, compute_log_probs)

    def _generator_completion(self, prompt: bytes, temperature: float):
        # Only runs on legacy openAI models that use old completion endpoints.
        self._reset_shared_data(prompt, temperature)  # update our shared data state

        try:
            prompt_decoded = prompt.decode("utf8")
            generator = self.client.completions.create(
                model=self.model_name,
                prompt=prompt_decoded,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1.0,  # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature,
                stream=True,
                logit_bias=self.logit_bias,
            )
            self.metrics.engine_input_tokens += len(self.tokenizer.encode(prompt))
        except Exception as e:
            # TODO: add retry logic, but keep token counts straight
            raise e

        for part in generator:
            if len(part.choices) > 0:
                chunk = part.choices[0].text or ""
            else:
                chunk = ""
            self.metrics.engine_output_tokens += len(
                self.tokenizer.encode(chunk.encode())
            )
            yield chunk.encode("utf8")

    def _generator_chat(self, prompt: bytes, temperature: float):
        # find the role tags
        pos = 0
        role_end = b"<|im_end|>\n"
        messages = []
        found = True
        input_token_count = 0

        # TODO: refactor this to method on parent class? (or a util function)
        while found:
            # find the role text blocks
            found = False
            for role_name, start_bytes in (
                ("system", b"<|im_start|>system\n"),
                ("user", b"<|im_start|>user\n"),
                ("assistant", b"<|im_start|>assistant\n"),
            ):
                if prompt[pos:].startswith(start_bytes):
                    pos += len(start_bytes)
                    end_pos = prompt[pos:].find(role_end)
                    if end_pos < 0:
                        assert (
                            role_name == "assistant"
                        ), "Bad chat format! Last role before gen needs to be assistant!"
                        break
                    btext = prompt[pos : pos + end_pos]
                    pos += end_pos + len(role_end)
                    message_content: str = btext.decode("utf8")
                    input_token_count += len(self.tokenizer.encode(btext))
                    messages.append({"role": role_name, "content": message_content})
                    found = True
                    break

        # Add nice exception if no role tags were used in the prompt.
        # TODO: Move this somewhere more general for all chat models?
        if messages == []:
            raise ValueError(
                f"The OpenAI model {self.model_name} is a Chat-based model and requires role tags in the prompt! \
            Make sure you are using guidance context managers like `with system():`, `with user():` and `with assistant():` \
            to appropriately format your guidance program for this type of model."
            )

        # Update shared data state
        self._reset_shared_data(prompt[:pos], temperature)

        # API call and response handling
        try:
            # Ideally, for the metrics we would use those returned by the
            # OpenAI API. Unfortunately, it appears that AzureAI hosted
            # models do not support returning metrics when streaming yet
            generator = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_streaming_tokens,
                n=1,
                top_p=1.0,  # TODO: this should be controllable like temp (from the grammar)
                temperature=temperature,
                stream=True,
                logit_bias=self.logit_bias,
            )
            self.metrics.engine_input_tokens += input_token_count

            for part in generator:
                if len(part.choices) > 0:
                    chunk = part.choices[0].delta.content or ""
                else:
                    chunk = ""
                encoded_chunk = chunk.encode("utf8")
                self.metrics.engine_output_tokens += len(
                    self.tokenizer.encode(encoded_chunk)
                )
                yield encoded_chunk

        except Exception as e:
            # TODO: add retry logic, keeping mind of token counts
            raise e

    def _generator(self, prompt: bytes, temperature: float):
        assert isinstance(prompt, bytes)
        if self.model_name in self._completion_models:
            return self._generator_completion(prompt, temperature)
        else:
            # Otherwise we are in a chat context
            return self._generator_chat(prompt, temperature)

    def set_logit_bias(self, logit_bias):
        self.logit_bias = logit_bias


class OpenAI(Grammarless):
    def __init__(
        self,
        model,
        tokenizer=None,
        echo=True,
        api_key=None,
        max_streaming_tokens=1000,
        timeout=0.5,
        compute_log_probs=False,
        **kwargs,
    ):
        """Build a new OpenAI model object that represents a model in a given state.

        Parameters
        ----------
        model : str
            The name of the OpenAI model to use (e.g. gpt-3.5-turbo).
        tokenizer : None or tiktoken.Encoding
            The tokenizer to use for the given model. If set to None we use `tiktoken.encoding_for_model(model)`.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        api_key : None or str
            The OpenAI API key to use for remote requests, passed directly to the `openai.OpenAI` constructor.
        max_streaming_tokens : int
            The maximum number of tokens we allow this model to generate in a single stream. Normally this is set very
            high and we rely either on early stopping on the remote side, or on the grammar terminating causing the
            stream loop to break on the local side. This number needs to be longer than the longest stream you want
            to generate.
        **kwargs :
            All extra keyword arguments are passed directly to the `openai.OpenAI` constructor. Commonly used argument
            names include `base_url` and `organization`
        """

        if client_class is None:
            raise Exception(
                "Please install the openai package version >= 1 using `pip install openai -U` in order to use guidance.models.OpenAI!"
            )

        super().__init__(
            engine=OpenAIEngine(
                tokenizer=tokenizer,
                api_key=api_key,
                max_streaming_tokens=max_streaming_tokens,
                timeout=timeout,
                compute_log_probs=compute_log_probs,
                model=model,
                **kwargs,
            ),
            echo=echo,
        )
