import os
import re
import textwrap
import warnings

from typing import Optional, Sequence, Union

try:
    import torch
except ModuleNotFoundError:
    pass


try:
    import transformers as transformers_package

    has_transformers = True
except ModuleNotFoundError:
    has_transformers = False


from .._model import Engine, Model
from .._tokenizer import Tokenizer

# Formed by comparing model and tokenizer from_pretrained methods
# transformers/models/auto/auto_factory.py
# transformers/models/auto/tokenization_auto.py
_COMMON_TRANSFORMERS_KWARGS = [
    "cache_dir",
    "force_download",
    "proxies",
    "resume_download",
    "revision",
    "subfolder",
    "trust_remote_code",
]

def load_transformers_model(model, **kwargs):
    # intantiate the model if needed
    if isinstance(model, str):

        # make sure transformers is installed
        if not has_transformers:
            raise Exception(
                "Please install transformers with `pip install transformers` in order to use guidance.models.Transformers!"
            )
        model = transformers_package.AutoModelForCausalLM.from_pretrained(model, **kwargs)
    return model

class ByteDecoderError(Exception):
    pass

class ByteTokensError(Exception):
    pass

class TransformersTokenizer(Tokenizer):
    def __init__(
        self,
        model: Union[str, "transformers_package.PreTrainedModel"],
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
            None,
        ],
        chat_template=None,
        ignore_bos_token=False,
        **kwargs,
    ):
        if transformers_tokenizer is None:
            if isinstance(model, str):
                transformers_tokenizer, byte_tokens = self._tokenizer(model, **kwargs)
            else:
                raise ValueError(
                    "A model object was passed in, but no tokenizer was provided. Please provide a tokenizer."
                )
        else:
            is_ptt = isinstance(transformers_tokenizer, transformers_package.PreTrainedTokenizer)
            is_ptt_fast = isinstance(
                transformers_tokenizer, transformers_package.PreTrainedTokenizerFast
            )
            assert is_ptt or is_ptt_fast
            byte_tokens = self._byte_tokens(transformers_tokenizer, **kwargs)

        self._orig_tokenizer = transformers_tokenizer

        # Chat Template logic
        if chat_template is None and hasattr(self._orig_tokenizer, "chat_template"):
            chat_template = self._orig_tokenizer.chat_template

        # the superclass does most of the work once we have the tokens
        super().__init__(
            byte_tokens,
            chat_template,
            None if ignore_bos_token else transformers_tokenizer.bos_token_id,
            transformers_tokenizer.eos_token_id,
        )

    def _tokenizer(self, model: str, **kwargs) -> tuple[
        Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ],
        list[bytes],
    ]:
        # make sure transformers is installed
        if not has_transformers:
            raise ImportError("Please install transformers with `pip install transformers`")

        try:
            tokenizer = transformers_package.AutoTokenizer.from_pretrained(
                model, use_fast=False, **kwargs
            )
            byte_tokens = self._byte_tokens(tokenizer, **kwargs)
        except ImportError:
            # Raise on ImportError because it's likely a missing dependency that the user can install
            raise
        except ByteTokensError as e:
            # Give a specific warning for ByteTokensError and fall back to fast tokenizer
            warnings.warn(f"Falling back to fast tokenizer. Could not build byte tokens for model {model!r} due to exception {e.__class__.__name__}: {e}")
        except Exception as e:
            # Fall back for other exceptions
            warnings.warn(f"Falling back to fast tokenizer. Could not load tokenizer for model {model!r} due to exception {e.__class__.__name__}: {e}")
        else:
            return tokenizer, byte_tokens

        tokenizer = transformers_package.AutoTokenizer.from_pretrained(
            model, use_fast=True, **kwargs
        )
        try:
            byte_tokens = self._byte_tokens(tokenizer)
        except ByteTokensError as e:
            raise ValueError(f"Fallback to fast tokenizer failed for model {model!r}") from e
        return tokenizer, byte_tokens

    def _byte_tokens(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ],
        **kwargs,
    ) -> list[bytes]:

        if hasattr(transformers_tokenizer, "byte_decoder"):
            try:
                self._check_byte_decoder(
                    transformers_tokenizer.byte_decoder, transformers_tokenizer
                )
            except ByteDecoderError as e:
                warnings.warn(
                    f"Tokenizer has a byte_decoder, but it can't be used to construct byte_tokens: {e}"
                )
                pass
            else:
                return self._byte_tokens_from_byte_decoder(transformers_tokenizer.byte_decoder, transformers_tokenizer)

        if hasattr(transformers_tokenizer, "sp_model"):
            return self._byte_tokens_from_sp_model(transformers_tokenizer)

        if kwargs.get("sp_whitespace", False):
            return self._byte_tokens_from_sp_whitespace(transformers_tokenizer)

        try:
            return self._byte_tokens_by_encoding_token_strings(transformers_tokenizer)
        except ValueError as e:
            warnings.warn(
                f"Could not build_byte tokens from the tokenizer by encoding token strings: {e}"
            )
            pass

        fallback_byte_decoder = self._fallback_byte_decoder()
        try:
            self._check_byte_decoder(
                fallback_byte_decoder, transformers_tokenizer
            )
        except ByteDecoderError as e:
            # Should be the only exception that is raised in _byte_tokens
            raise ByteTokensError(
                "Could not build byte tokens from the tokenizer, and falling back to a standard gpt2 byte_decoder failed"
            ) from e
        return self._byte_tokens_from_byte_decoder(fallback_byte_decoder, transformers_tokenizer)

    def _byte_tokens_from_byte_decoder(
        self,
        byte_decoder: dict[str, int],
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        for i in range(len(transformers_tokenizer)):
            byte_coded = bytes(
                [byte_decoder[c] for c in transformers_tokenizer.convert_ids_to_tokens(i)]
            )
            byte_tokens[i] = byte_coded
        return byte_tokens

    def _byte_tokens_from_sp_model(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ]
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        special_tokens_map = {
            id: token for token, id in transformers_tokenizer.get_added_vocab().items()
        }
        space_prefix = "▁".encode()
        for i in range(len(transformers_tokenizer)):
            if i in special_tokens_map:
                byte_coded = special_tokens_map[i].encode()
            else:
                byte_coded = re.sub(
                    rb"<0x(..)>",
                    lambda x: bytes.fromhex(x[1].decode()),
                    transformers_tokenizer.sp_model.id_to_piece(i).encode(),
                )
            byte_tokens[i] = byte_coded.replace(space_prefix, b" ")
        return byte_tokens

    def _byte_tokens_from_sp_whitespace(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ]
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        if hasattr(transformers_tokenizer, "get_vocab"):
            space_prefix = "▁".encode()
            vocab = transformers_tokenizer.get_vocab()
            for token, tok_id in vocab.items():
                byte_coded = token.encode()
                byte_tokens[tok_id] = byte_coded.replace(space_prefix, b" ")
        else:
            raise ValueError("Tokenizer does not have a get_vocab method")
        return byte_tokens

    def _byte_tokens_by_encoding_token_strings(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        special_tokens_map = {
            id: token for token, id in transformers_tokenizer.get_added_vocab().items()
        }
        byte_encoder = self._bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        for i in range(len(transformers_tokenizer)):
            if i in special_tokens_map:
                byte_coded = special_tokens_map[i].encode()
            else:
                token = transformers_tokenizer.convert_ids_to_tokens(i)
                if isinstance(token, bytes):
                    byte_coded = token
                elif isinstance(token, str):
                    if hasattr(transformers_tokenizer, "convert_tokens_to_string"):
                        token_str = transformers_tokenizer.convert_tokens_to_string([token])
                        encoded_str = transformers_tokenizer.encode(token_str)
                        if len(encoded_str) != 1:
                            raise ValueError(f"Round-trip encoding of tokens [{token}] failed! Got {encoded_str}")
                        roundtrip_id = encoded_str[0]
                        if roundtrip_id == i:
                            byte_coded = token_str.encode()
                        else:
                            byte_coded = bytes([byte_decoder[c] for c in token])
                    else:
                        byte_coded = token.encode()
                else:
                    raise ValueError(f"Unexpected token type: {type(token)}")
            byte_tokens[i] = byte_coded
        return byte_tokens

    def _fallback_byte_decoder(self) -> dict[str, int]:
        byte_decoder = transformers_package.AutoTokenizer.from_pretrained(
            "gpt2", use_fast=False
        ).byte_decoder # fall back to gpt2 mapping

        # some special tokens may not have their whitespace encoded...
        byte_decoder[" "] = 32
        byte_decoder["\n"] = 10
        byte_decoder["\r"] = 13
        byte_decoder["\t"] = 9
        byte_decoder["▁"] = 32

        return byte_decoder

    def _check_byte_decoder(
        self,
        byte_decoder: dict[str, int],
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
        ],
    ) -> None:

        def check_byte_decoder_has_all_bytes() -> None:
            # This is here because some tokenizers are bad and don't have all the bytes (I'm looking at you, microsoft/phi2)
            all_bytes = set()
            for x in transformers_tokenizer.get_vocab().keys():
                for y in x:
                    all_bytes.add(y)
            if not set(byte_decoder.keys()) >= all_bytes:
                raise ByteDecoderError(
                    f"Byte decoder is missing bytes: {all_bytes - set(byte_decoder.keys())}"
                )

        def check_byte_decoder_complex_round_trip() -> None:
            # run a quick spot check to verify we can rebuild complex multi-token unicode symbols
            s = "’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨"
            reconstructed = b""
            try:
                input_ids = transformers_tokenizer(s)["input_ids"]
                for i in input_ids:
                    nxt_bytes = []
                    token_str = transformers_tokenizer.convert_ids_to_tokens(i)
                    for c in token_str:
                        nxt_bytes.append(byte_decoder[c])
                    reconstructed += bytes(nxt_bytes)
                # Check if the tokenizer has a bos_token attribute, and if it does, check
                # if it's at the start of the reconstructed bytes
                # Some tokenizers add this automatically as part of the call function, so
                # we need to remove it to compare
                if hasattr(transformers_tokenizer, "bos_token") and transformers_tokenizer.bos_token and reconstructed.startswith(
                    transformers_tokenizer.bos_token.encode()
                ):
                    reconstructed = reconstructed[len(transformers_tokenizer.bos_token) :]
            # TODO: can we narrow this exception?
            except Exception as e:
                msg = textwrap.dedent(
                    f"""
                    The tokenizer being used is unable to convert a special character in {s}.
                    For models with sentencepiece based tokenizers (e.g. llama, phi-3-mini),
                    installing sentencepiece often fixes this issue (pip install sentencepiece).
                    """
                )
                raise ByteDecoderError(msg) from e
            if reconstructed.decode() != s:
                raise ByteDecoderError(
                    f"Failed to reconstruct the string {s} from the tokenizer's byte_decoder: {reconstructed.decode()!r} != {s!r}"
                )

        check_byte_decoder_has_all_bytes()
        check_byte_decoder_complex_round_trip()

    def _bytes_to_unicode(self):
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def encode(self, byte_string: bytes) -> list[int]:
        assert isinstance(byte_string, bytes)
        # HF tokenizers take in strings apparently
        tokenization = self._orig_tokenizer(byte_string.decode(), add_special_tokens=False)
        return tokenization["input_ids"]

    def decode(self, tokens: Sequence[int]) -> bytes:
        decoded_str = self._orig_tokenizer.decode(tokens)
        return decoded_str.encode()

    def recode(self, tokens: Sequence[int]) -> list[int]:
        # the encode/decode cycle might not work if we have partial unicode strings
        used_tokens = len(tokens)
        for _ in range(3):
            try:
                first_decode = self.decode(tokens).decode("utf8")
            except UnicodeDecodeError:
                if used_tokens == 0:
                    break
                else:
                    used_tokens -= 1

        new_ids = list(self.encode(first_decode.encode("utf-8")))
        if used_tokens < len(tokens):
            new_ids += tokens[used_tokens:]

        # HACK: check for a bug in the HuggingFace tokenizer
        # (that will just add extra spaces during an encode-decode cycle)
        second_decode = self._orig_tokenizer.decode(new_ids)
        if (
            second_decode != first_decode
            and len(second_decode) == len(first_decode) + 1
            and second_decode.startswith("<s>  ")
        ):
            new_ids = new_ids[0:1] + new_ids[2:]

        return new_ids


class TransformersEngine(Engine):
    def __init__(self, model, tokenizer, compute_log_probs: bool, chat_template=None, **kwargs):
        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser("~/.transformers_model"), "r") as file:
                    model = file.read().replace("\n", "")
            except:
                pass

        self.model_obj = load_transformers_model(model, **kwargs)

        if not isinstance(model, str):
            self.model = model.__class__.__name__
        self.device = self.model_obj.device  # otherwise note the current device

        self._past_key_values = None
        self._cached_logits = None
        self._cached_token_ids: list[int] = []

        # Set attr for malformed tokenizer hack.
        # If more models start doing this, generalize into a util function.
        if hasattr(self.model_obj.config, "model_type"):
            if self.model_obj.config.model_type in ["phi3"]:
                self._disable_retokenize_check = True

        # Automatically fill common args between Transformers
        # model and tokenizer
        passed_common_kwargs = {}
        for arg_name in _COMMON_TRANSFORMERS_KWARGS:
            if arg_name in kwargs:
                passed_common_kwargs[arg_name] = kwargs[arg_name]

        # Create the tokenizer
        my_tokenizer = TransformersTokenizer(
            model, tokenizer, chat_template, **passed_common_kwargs
        )

        super().__init__(
            my_tokenizer,
            compute_log_probs=compute_log_probs,
        )

    def get_logits(self, prompt: bytes, token_ids: list[int], media: Optional[dict] = None):
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """

        # make sure we don't run off the end of the model
        if len(token_ids) >= getattr(self.model_obj.config, "max_position_embeddings", 1e10):
            raise Exception(
                f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!"
            )

        # get the number of cache positions we are using
        cache_token_ids = self._cached_token_ids
        num_cached = 0
        for id in cache_token_ids:
            if (
                num_cached >= len(cache_token_ids)
                or num_cached >= len(token_ids)
                or token_ids[num_cached] != id
            ):
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._past_key_values
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if past_length > num_cached:
            # note we recompute the last token because we don't bother to handle the special case of just computing logits
            past_length = max(0, num_cached - 1)
            self._past_key_values = tuple(
                tuple(p[..., :past_length, :] for p in v) for v in past_key_values
            )
        cache_token_ids[past_length:] = []

        # call the model
        new_token_ids = token_ids[past_length:]
        if len(new_token_ids) > 0:
            with torch.no_grad():
                # Not all models support batched tokens for some reason
                try:
                    model_out = self.model_obj(
                        input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                        past_key_values=self._past_key_values,
                        use_cache=True,
                        position_ids=torch.arange(past_length, past_length + len(new_token_ids))
                        .unsqueeze(0)
                        .to(self.device),
                        attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(
                            self.device
                        ),
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                except AssertionError:
                    for i, new_token_id in enumerate(new_token_ids):
                        input_ids = torch.tensor([new_token_id]).unsqueeze(0).to(self.device)

                        model_out = self.model_obj(
                            input_ids=input_ids,
                            past_key_values=self._past_key_values,
                            use_cache=True,
                            position_ids=torch.arange(past_length, past_length + 1)
                            .unsqueeze(0)
                            .to(self.device),
                            attention_mask=torch.ones(1, past_length + 1).to(self.device),
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                        self._past_key_values = model_out.past_key_values
                        past_length += 1

            # save the results
            self._past_key_values = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            # Need to add special truncating logic here for weird models that have a different output size than tokenizer vocab
            self._cached_logits = (
                model_out.logits[0, -1, : len(self.tokenizer.tokens)].cpu().numpy()
            )
            self.metrics.engine_input_tokens += len(new_token_ids)
            self.metrics.engine_output_tokens += 1

        return self._cached_logits


class Transformers(Model):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        **kwargs,
    ):
        """Build a new Transformers model object that represents a model in a given state."""
        super().__init__(
            TransformersEngine(
                model,
                tokenizer,
                compute_log_probs,
                chat_template=chat_template,
                **kwargs,
            ),
            echo=echo,
        )

