# TODO(nopdive): This module requires a memory review.

import copy
import logging
import queue
import re
import threading
import weakref

import time
from asyncio import CancelledError
from typing import Iterator, Optional, TYPE_CHECKING, Callable
from multiprocessing import Manager, Process
from typing import Any, Union
from enum import Enum
import psutil

import numpy as np

from ..trace import (
    NodeAttr,
    StatelessGuidanceInput,
    StatefulGuidanceInput,
    LiteralInput,
    EmbeddedInput,
    RoleOpenerInput,
    RoleCloserInput,
    TextOutput,
    CaptureOutput,
    TraceHandler,
)
from ..trace._trace import AudioOutput, AudioInput, VideoInput, VideoOutput
from ..visual import (
    TraceMessage,
    AutoRenderer,
    trace_node_to_str,
    trace_node_to_html,
    GuidanceMessage,
    Renderer,
    ExecutionStartedMessage,
    ExecutionCompletedMessage,
    TokensMessage,
    MetricMessage,
    OutputRequestMessage,
    JupyterWidgetRenderer,
)
from ..visual._async import run_async_coroutine, async_task

try:
    from IPython.display import clear_output, display, HTML

    ipython_is_imported = True
except ImportError:
    ipython_is_imported = False

logger = logging.getLogger(__name__)

from .._schema import (
    EngineCallResponse,
    EngineOutput,
    GenToken,
    GuidanceEngineMetrics,
    VisBytesChunk,
    GenTokenExtra,
)
from .._utils import softmax, CaptureEvents, log_cleanup, log_init, log_copy, to_utf8_or_bytes_string
from .._parser import TokenParser
from .._grammar import (
    Function, # for da types, just for you Hudson <3 
    GrammarFunction,
    string,
    _call_pool,
    _tag_pattern,
    Null,
    replace_model_variables,
    unreplace_model_variables,
    select,
)
from ._tokenizer import Tokenizer

if TYPE_CHECKING:
    from ..library._block import ContextBlock

# define some constants we will reuse many times
_null_grammar = string("")
format_pattern = re.compile(r"<\|\|_.*?_\|\|>", flags=re.DOTALL)


class PeriodicMetricsGenerator:
    def __init__(self, renderer: Renderer, monitor: "Monitor", sleep_sec=0.5):
        self._renderer = renderer
        self._monitor = monitor
        self._sleep_sec = sleep_sec
        self._task = None
        self._task_cancelled = False
        self._is_paused = False
        run_async_coroutine(self._emit())

    def start(self):
        self._task = run_async_coroutine(async_task(self._emit())).result()

    def stop(self):
        if self._task is not None:
            self._task.cancel()
            # TODO: it seems _task.cancel() is not working, use a flag to stop the task
            self._task_cancelled = True

    def pause(self):
        """
        Pauses the model by setting the internal _is_paused flag to True.
        
        This method can be used to temporarily halt the model's operations.
        """
        self._is_paused = True

    def resume(self):
        """
        Resume the model's operation by setting the paused state to False.

        This method changes the internal state of the model to indicate that it is no longer paused.
        """
        self._is_paused = False

    async def _emit(self):
        import asyncio
        import time

        time_start = time.time()
        while not self._task_cancelled:
            try:
                await asyncio.sleep(self._sleep_sec)

                cpu_percent = self._monitor.get_metric(MonitoringMetric.CPU_USAGE)
                mem_percent = self._monitor.get_metric(MonitoringMetric.MEM_USAGE)
                gpu_percent = self._monitor.get_metric(MonitoringMetric.GPU_USAGE)
                gpu_used_vram = self._monitor.get_metric(MonitoringMetric.GPU_USED_MEM)

                if gpu_percent:
                    gpu_percent = max(gpu_percent)
                else:
                    gpu_percent = 0

                if gpu_used_vram:
                    gpu_used_vram = max(gpu_used_vram)
                else:
                    gpu_used_vram = 0

                if not cpu_percent:
                    cpu_percent = 0

                if not mem_percent:
                    mem_percent = 0

                time_end = time.time()
                time_elapsed = time_end - time_start

                if not self._is_paused:
                    self._renderer.update(MetricMessage(name="wall time", value=time_elapsed))
                    self._renderer.update(MetricMessage(name="cpu", value=cpu_percent))
                    self._renderer.update(MetricMessage(name="ram", value=mem_percent))
                    self._renderer.update(MetricMessage(name="gpu", value=gpu_percent))
                    self._renderer.update(MetricMessage(name="vram", value=gpu_used_vram))
            except CancelledError:
                logger.debug("METRICGEN:canceling")
                break
            except Exception as e:
                logger.debug(f"METRICGEN: {repr(e)}")
                break

        logger.debug("METRICGEN:exiting")


class PostExecMetrics:
    def __init__(self, renderer: Renderer, monitor: "Monitor"):
        self._renderer = renderer
        self._monitor = monitor

    def emit_messages(self, lm: "Model"):
        token_reduction = self._monitor.get_metric(MonitoringMetric.TOKEN_REDUCTION, lm)
        if token_reduction is not None:
            self._renderer.update(
                MetricMessage(
                    name="token reduction",
                    value=token_reduction * 100,
                )
            )

        output_tokens = self._monitor.get_metric(MonitoringMetric.OUTPUT_TOKENS, lm)
        if output_tokens is not None:
            self._renderer.update(MetricMessage(name="consumed", value=output_tokens))

        avg_latency = self._monitor.get_metric(MonitoringMetric.AVG_LATENCY, lm)
        if avg_latency is not None:
            self._renderer.update(MetricMessage(name="avg latency", value=avg_latency))


def _engine_cleanup(renderer: Renderer, msg_recv: Callable[[GuidanceMessage], None], 
                    monitor: Optional['Monitor'],  
                    periodic_metrics_generator: Optional[PeriodicMetricsGenerator],
                    log_msg: str):
    renderer.unsubscribe(msg_recv)

    try:
        # force renderer cleanup
        # TODO: figure out why in some cases _recv_task and _send_task are not stopped
        from ..visual._renderer import _cleanup
        if isinstance(renderer, AutoRenderer) and isinstance(renderer._renderer, JupyterWidgetRenderer):
            _cleanup(renderer._renderer._recv_queue, renderer._renderer._send_queue, f"renderer({id(renderer)})")
    except Exception as e:
        logger.error(f"Failed to force-cleanup renderer: {e}")

    if periodic_metrics_generator is not None:
        try:
            periodic_metrics_generator.stop()
        except Exception as e:
            logger.error(f"Failed to stop periodic metrics generator: {e}")

    if monitor is not None:
        try:
            monitor.stop()
        except Exception as e:
            logger.error(f"Failed to stop monitor: {e}")

    log_cleanup(log_msg)


def _wrapped_msg_recv(engine_weak_ref: weakref.ref) -> Callable[[GuidanceMessage], None]:
    def closure(message):
        return _msg_recv(engine_weak_ref, message)

    return closure


def _msg_recv(engine_weakref: weakref.ReferenceType, message: GuidanceMessage) -> None:
    engine = engine_weakref()
    if engine is None:
        return

    # NOTE(nopdive): This is usually run on a background thread.
    logger.debug(f"ENGINE:msg_recv:{message}")
    if isinstance(message, ExecutionStartedMessage):
        # TODO(nopdive): Start execution logic here.
        logger.debug(f"ENGINE:msg_recv:START STUB")
        if engine.periodic_metrics_generator is not None:
            engine.periodic_metrics_generator.resume()
    elif isinstance(message, ExecutionCompletedMessage) and message.is_err:
        pass
    elif isinstance(message, (ExecutionCompletedMessage, OutputRequestMessage)):
        # print("last_state")
        if isinstance(message, TokensMessage):
            last_model: "Model" = engine.model_dict[message.last_trace_id]
            last_trace_id = message.last_trace_id
        else:
            last_model = list(engine.model_dict.values())[-1]
            last_trace_id = last_model._id

        failed = False
        processed_gen_tokens: list[GenTokenExtra] = []  # suppress IDE warnings by definition
        try:
            processed_gen_tokens = last_model.get_per_token_stats()
        except Exception as e:
            logger.error(f"Failed to get per token stats: {e}")
            failed = True

        if not failed:
            final_text = "".join([gen_token.text for gen_token in processed_gen_tokens])
            logger.debug(f"ENGINE:final_text:{final_text}")
            logger.debug(f"ENGINE:model_state:{last_model._state}")
            logger.debug(f"ENGINE:final_text == _state:{final_text == last_model._state}")

            tokens = [gen_token.token_id for gen_token in processed_gen_tokens]
            engine.renderer.update(
                TokensMessage(
                    trace_id=last_trace_id,
                    text=engine.tokenizer.decode(tokens).decode("utf-8"),
                    tokens=processed_gen_tokens,
                )
            )
            engine.renderer.update(MetricMessage(name="status", value="✓"))
        else:
            engine.renderer.update(MetricMessage(name="status", value="⚠"))

        if engine.periodic_metrics_generator is not None:
            engine.periodic_metrics_generator.pause()

        try:
            # send stats to the renderer
            engine.post_exec_metrics.emit_messages(last_model)
        except:
            pass


class Engine:
    """The engine owns the inference computation and is used/created by the Model class.

    Engine objects represent the expensive parts of inference. While Model objects are cheap and do not
    need to know about the tokenizer or the model parameters, Engine objects know about both. Many
    Model objects can reference a single Engine object. Engine objects can also be hidden behind a
    Server so a single server can serve many clients' model objects through a single Engine object.
    """

    def __init__(self, tokenizer: Tokenizer, 
                 compute_log_probs=False, 
                 enable_backtrack=True, 
                 enable_ff_tokens=True, 
                 renderer=None, 
                 enable_monitoring=True,
                 **kwargs):
        # TODO(nopdive): Hook up renderer keyword to all engines.
        self.tokenizer = tokenizer
        self.compute_log_probs = compute_log_probs
        self._enable_backtrack = enable_backtrack
        self._enable_ff_tokens = enable_ff_tokens
        self._enable_monitoring = enable_monitoring
        self._top_k = kwargs.get("top_k", 5)
        self.metrics = GuidanceEngineMetrics()

        if renderer is None:
            self.trace_handler = TraceHandler()
            self.renderer = AutoRenderer(self.trace_handler)
            # self.renderer = JupyterWidgetRenderer(self.trace_handler)
        else:
            self.renderer = renderer
            self.trace_handler = renderer._trace_handler

        msg_recv = _wrapped_msg_recv(weakref.ref(self))
        self.renderer.subscribe(msg_recv)

        self.model_dict: weakref.WeakValueDictionary[int, Model] = weakref.WeakValueDictionary()

        self.monitor = None
        self.periodic_metrics_generator = None
        self.post_exec_metrics = None
        if self._enable_monitoring:
            self.monitor = Monitor(self.metrics)
            self.monitor.start()

            self.periodic_metrics_generator = PeriodicMetricsGenerator(self.renderer, self.monitor)
            self.periodic_metrics_generator.start()
            self.post_exec_metrics = PostExecMetrics(self.renderer, self.monitor)

        weakref.finalize(self, _engine_cleanup, self.renderer, msg_recv, self.monitor, self.periodic_metrics_generator, f"engine({id(self)})")
        log_init(f"engine({id(self)})")

    # These need to be properties because once an Engine is started, you can't change their behavior.
    @property
    def enable_backtrack(self):
        return self._enable_backtrack
    
    @property
    def enable_ff_tokens(self):
        return self._enable_ff_tokens
    
    @property
    def enable_monitoring(self):
        return self._enable_monitoring

    def get_chat_template(
        self,
    ):  # TODO [HN]: Add more logic here...should we instantiate class here? do we even need to?
        return (
            self.tokenizer.chat_template()
        )  # Instantiate the class before returning to client for now

    def reset_metrics(self):
        self.metrics = GuidanceEngineMetrics()

    def start(self, prompt, grammar, ensure_bos_token=True) -> TokenParser:
        # def __call__(self, grammar, max_tokens=1000000, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        # assert n == 1, "Still need to add support for n > 1!"

        # TODO: re-enable this? llguidance currently doesn't support model variables
        # note we only support a fixed set of engine variables for the sake of security
        # self._replacements = replace_model_variables(
        #     grammar, self, allowed_vars=["eos_token", "bos_token"]
        # )

        # right now we only support a text/bytes prompt parser state, so we extract that
        if isinstance(prompt, bytes):
            prompt = prompt
        elif isinstance(prompt, str):
            prompt = bytes(prompt, encoding="utf8")
        elif isinstance(prompt, TokenParser):
            raise NotImplementedError(
                "Still need to implement support for extending a full Parser trace."
            )
        else:
            raise Exception("The passed prompt is of an unknown type!")

        return TokenParser(
            grammar=grammar,
            tokenizer=self.tokenizer,
            prompt=prompt,
            ensure_bos_token=ensure_bos_token,
            enable_backtrack=self.enable_backtrack,
            enable_ff_tokens=self.enable_ff_tokens,
        )

    def __call__(
            self, 
            prompt: Union[str, TokenParser], 
            grammar: Function,
            ensure_bos_token: bool = True,
            echo: bool = True,
        ) -> Iterator[EngineCallResponse]:
        """Main entry point for the inference-parser loop. Yields EngineCallResponse objects as
        the parser advances through the grammar.

        Parameters
        ----------
        prompt : str or Parser
            This is represents the current state of a guidance parser that will be extended
            using the passed grammar. If a string is given then we assume the previous parser
            state is just a fixed string prompt, if a full Parser is given then we extend that
            parser by appending the new grammar to the parser's current grammar and then
            inferencing the model. (TODO: implement full parser extension support)
        grammar: Function
            Grammar (RawFunction or GrammarFunction) used to extend the prompt.
        ensure_bos_token: bool
            Ensures that the prompt ends with the BOS token.
        """
        parser = self.start(prompt, grammar, ensure_bos_token)

        has_get_logits = True
        engine_output = None
        logits_lat_ms = 0
        delayed_bytes = b""
        delayed_engine_outputs: list[EngineOutput] = []
        while not parser.done():
            t0 = time.time()

            tokens, mask_fut, backtrack = parser.advance(engine_output)

            # Note that has_pending_stop implies that the response is a stop response,
            # but the converse is not true. We can therefore avoid some (but not all)
            # unnecessary calls to get_logits on the final iteration.
            has_pending_stop = parser.has_pending_stop()

            if has_get_logits and not has_pending_stop:
                try:
                    t0 = time.time()
                    logits = self.get_logits(token_ids=tokens)
                    logits_lat_ms = (time.time() - t0) * 1000
                except NotImplementedError:
                    # Permanently fall-back to get_next_token if get_logits is not implemented
                    has_get_logits = False
                    logits = None
                    logits_lat_ms = 0
            else:
                logits = None

            # Important: don't wait on this future until after getting the logits;
            # this allows the mask to be built concurrently with model inference
            mask, ll_response = mask_fut.result()

            engine_response = ll_response.progress.to_engine_call_response()
            engine_response.backtrack = backtrack
            if engine_output:
                engine_response.engine_outputs.append(engine_output)

            # NOTE (loc): Temporary solution to quickly check which segments are generated and which are force-forwarded to animate visualizations on the UI
            # These tokens in chunk will not be used for final visualization
            # TODO: This should be handled by the interpreter
            if echo and engine_response.new_bytes:
                try:
                    _new_bytes = delayed_bytes + engine_response.new_bytes
                    _tokens = parser.tokenizer.encode(_new_bytes)
                    delayed_bytes = b""
                except UnicodeDecodeError:
                    # similar to what we did in _run_stateless function, if we could not decode current bytes
                    # we will delay until we can decode them
                    delayed_bytes += engine_response.new_bytes
                    if engine_output:
                        engine_response.engine_outputs.pop()
                        delayed_engine_outputs.append(engine_output)

                if not delayed_bytes:
                    ff_token_start_idx = 1
                    if engine_output is None and len(delayed_engine_outputs) == 0:
                        ff_token_start_idx = 0
                    elif engine_output.issued_token.token_id == _tokens[0] and len(delayed_engine_outputs) == 0:
                        # this is generated
                        engine_response.generated_bytes = parser.tokenizer.decode([_tokens[0]])
                        engine_output.issued_token.is_generated = True
                        engine_response.generated_tokens.append(engine_output.issued_token)
                    else:
                        # handle delayed bytes
                        engine_outputs = delayed_engine_outputs + [engine_output] if engine_output else []
                        engine_output_tokens = [e.issued_token.token_id for e in engine_outputs]

                        generated = to_utf8_or_bytes_string(parser.tokenizer.decode(engine_output_tokens))
                        force_forwarded = _new_bytes.decode("utf-8")

                        if force_forwarded.startswith(generated):
                            engine_output_tokens = np.array(engine_output_tokens)
                            ff_tokens = np.array(_tokens)

                            # check if engine_output_tokens in ff_tokens
                            _idx = -1
                            for _i in range(0, len(ff_tokens) - len(engine_output_tokens) + 1):
                                if np.array_equal(engine_output_tokens, ff_tokens[_i:_i+len(engine_output_tokens)]):
                                    _idx = _i + len(engine_output_tokens)
                                    break

                            if _idx < 0:
                                ff_token_start_idx = 0
                            else:
                                # all previous tokens before _idx are generated
                                engine_response.generated_bytes = parser.tokenizer.decode(ff_tokens[:_idx])
                                idx_in_engine_output_tokens = 0
                                for _i in range(_idx):
                                    matching_engine_output = None
                                    if _tokens[_i] == engine_output_tokens[idx_in_engine_output_tokens]:
                                        matching_engine_output = engine_outputs[idx_in_engine_output_tokens]
                                        idx_in_engine_output_tokens += 1
                                    engine_response.generated_tokens.append(
                                        GenToken(
                                            token_id=_tokens[_i],
                                            prob=1.0 if not matching_engine_output else matching_engine_output.issued_token.prob,
                                            text=parser.tokenizer.decode([_tokens[_i]]) if not matching_engine_output else matching_engine_output.issued_token.text,
                                            latency_ms=0.0 if not matching_engine_output else matching_engine_output.issued_token.latency_ms,
                                            is_generated=True,
                                        )
                                    )
                                ff_token_start_idx = _idx
                        else:
                            ff_token_start_idx = 0

                    if len(_tokens[ff_token_start_idx:]):
                        engine_response.force_forwarded_bytes = parser.tokenizer.decode(
                            _tokens[ff_token_start_idx:]
                        )
                        for _token in _tokens[ff_token_start_idx:]:
                            engine_response.force_forwarded_tokens.append(
                                GenToken(
                                    token_id=_token,
                                    prob=1.0,
                                    text=to_utf8_or_bytes_string(parser.tokenizer.decode([_token])),
                                    latency_ms=0,
                                    is_force_forwarded=True,
                                )
                            )

                    delayed_engine_outputs = []
            elif not echo and engine_response.new_bytes:
                # do not collect tokens-metrics if echo is disabled
                engine_response.generated_bytes = engine_response.new_bytes
                engine_response.generated_tokens.clear()

            # process engine_response
            # NOTE (loc): We should not yield the engine_response if new_bytes are invalid utf-8 bytes
            # delayed bytes should be handled here in the engine
            yield engine_response

            if ll_response.stop:
                assert mask is None
                # May raise an exception if the parser is in an bad state!
                parser.cleanup()
                # Ensure we break AFTER yielding the final response
                break

            # If there was a pending stop, we should have broken out of the loop
            assert not has_pending_stop

            # Help the type checker: assert that everything we need to get the next token is not None
            assert mask is not None
            assert ll_response.temperature is not None

            can_finish_early = parser.is_accepting() and self.tokenizer.eos_token_id is not None

            if can_finish_early:
                # Type checker needs some help
                assert self.tokenizer.eos_token_id is not None
                # Should be equivalent to parser.is_accepting()
                assert mask[self.tokenizer.eos_token_id]
                # Whenever we are in an accepting state, we will allow the model to generate whatever it wants
                # but we will treat any "illegal" tokens as EOS, allowing the model to finish gracefully.
                # Hence, mask must be None
                mask_for_sampling = None
            else:
                mask_for_sampling = mask

            engine_output = self.get_next_token_with_top_k(
                logits=logits,
                logits_lat_ms=logits_lat_ms,
                token_ids=tokens,
                mask=mask_for_sampling,
                temperature=ll_response.temperature,
                k=self._top_k,
                force_return_unmasked_probs=echo
            )

            if can_finish_early and not mask[engine_output.issued_token.token_id]:
                # Type checker needs some help
                assert self.tokenizer.eos_token_id is not None
                engine_output.issued_token.token_id = self.tokenizer.eos_token_id

    def get_next_token_with_top_k(
        self,
        logits: Optional[np.ndarray],
        logits_lat_ms: Optional[float],
        token_ids: list[int],
        mask: Optional[bytes],
        temperature: float,
        k: int = 5,
        force_return_unmasked_probs: bool = False,
    ) -> EngineOutput:
        """Get the next token and associated top-k tokens from the engine.

        Parameters
        -------
        logits : Optional[np.ndarray]
            The logits for the current token ids in the sequence.
            If None, the model will call get_logits to get the logits.
        logits_lat_ms: Optional[float]
            The time taken to compute the logits.
            If logits is None, the model will call get_logits to measure the time.
        token_ids : list[int]
            The current token ids in the sequence.
        mask : Optional[bytes]
            The mask to apply to the logits.
        temperature : float
            The temperature to apply to the logits.
        k : int
            The number of top-k tokens to return.
        force_return_unmasked_probs: bool
            If True, the top-k unmasked probabilities will be returned.

        Returns
        -------
        EngineOutput
            The output from the model.
        """

        if logits is None:
            t0 = time.time()
            try:
                logits = self.get_logits(token_ids)
            except NotImplementedError:
                # fallback to orignal get_next_token method
                _t0 = time.time()
                token_id = self.get_next_token(
                    token_ids=token_ids,
                    mask=mask,
                    temperature=temperature,
                )
                _lat = (time.time() - _t0) * 1000

                _issued_token = GenToken(
                    token_id=token_id,
                    prob=1.0,
                    text=to_utf8_or_bytes_string(self.tokenizer.decode([token_id])),
                    latency_ms=_lat,
                    is_generated=True,
                )

                return EngineOutput(
                    issued_token=_issued_token,
                    top_k=[_issued_token],
                    masked_top_k=[_issued_token] if mask is not None else None,
                )
            lat_ms = (time.time() - t0) * 1000
        else:
            lat_ms = logits_lat_ms

        def get_top_k(_probs: np.ndarray, _k: int = 5) -> list[GenToken]:
            top_k_indices = np.argsort(_probs)[::-1][:_k]
            top_k_probs = _probs[top_k_indices]

            return [
                GenToken(
                    token_id=token,
                    prob=prob,
                    text=to_utf8_or_bytes_string(self.tokenizer.decode([token])),
                    latency_ms=lat_ms,
                    is_generated=True,
                )
                for token, prob in zip(top_k_indices, top_k_probs)
                if prob > 0
            ]

        # compute top-k without masking
        probs = (
            softmax(np.array(logits))
            if temperature < 0.0001
            else softmax(np.array(logits) / temperature)
        )

        top_k: list[GenToken] = []
        if force_return_unmasked_probs:
            top_k = get_top_k(probs, k)

        # compute top-k with masking
        masked_top_k: list[GenToken] = []
        if mask is not None:
            # shift logits to [0 - max] range first and apply mask
            masked_logits = (logits - np.min(logits)) * np.frombuffer(mask, dtype=np.uint8)
            masked_probs = (
                softmax(masked_logits)
                if temperature < 0.0001
                else softmax(masked_logits / temperature)
            )
            masked_top_k = get_top_k(masked_probs, k)

        if temperature < 0.0001:
            if len(masked_top_k) > 0:
                issued_token = masked_top_k[0]
            else:
                if len(top_k) == 0:
                    top_k = get_top_k(probs, k)
                issued_token = top_k[0]
        else:
            # we need to sample from the probabilities
            if mask is None:
                sampled_index = np.random.choice(len(probs), p=probs)
                sampled_prob = probs[sampled_index]
            else:
                sampled_index = np.random.choice(len(masked_probs), p=masked_probs)
                sampled_prob = masked_probs[sampled_index]

            issued_token = GenToken(
                token_id=sampled_index,
                prob=sampled_prob,
                text=to_utf8_or_bytes_string(self.tokenizer.decode([sampled_index])),
                latency_ms=lat_ms,
                is_generated=True,
            )

        output = EngineOutput(
            issued_token=issued_token,
            top_k=top_k,
            masked_top_k=None if not masked_top_k else masked_top_k,
            is_backtracked=False,
        )

        return output

    def get_next_token(
        self, token_ids: list[int], mask: Optional[bytes], temperature: float
    ) -> int:
        # Prefer to implement get_logits over get_next_token as it allows for concurrent mask computation
        raise NotImplementedError

    def get_logits(self, token_ids: list[int]) -> np.ndarray:
        # Prefer to implement get_logits over get_next_token as it allows for concurrent mask computation
        raise NotImplementedError

    def get_per_token_topk_probs(self, token_ids: list[int], top_k: int = 5) -> list[GenToken]:
        """Get the top-k probabilities for each token in the sequence."""
        raise NotImplementedError

    def sample_with_temperature(
        self, logits: np.ndarray, mask: Optional[bytes], temperature: float
    ) -> int:
        if mask is not None:
            logits += np.frombuffer(mask, dtype=np.uint8)
        if temperature < 0.0001:
            return int(np.argmax(logits))
        # Get probabilities from softmax
        probabilities = softmax(logits / temperature)
        # Sample an index based on the probabilities
        sampled_index = np.random.choice(len(logits), p=probabilities)
        return sampled_index

    def _report_failed_match(self, prompt):
        """Note that this can be overridden by subclasses that have more likely reasons than a bug in the token set (like remote models)."""
        return Exception(
            "We can't consume any more tokens, but we are not yet done! Perhaps your model's token set is incomplete? This happened after the prompt:"
            + str(prompt[-40:])
        )


_id_counter = 0  # Counter for identifiers, this has to be outside the model to handle child classes properly.


class Model:
    """The base guidance model object, which represents a model in a given state.

    Model objects are immutable representations of model state, so whenever you change
    them you get a new Model object. However, these copies share the "expensive"
    parts of the underlying model like the parameters and KV-cache, through a shared
    Engine, so making copies of Model objects is cheap.

    .. automethod:: __add__
    """

    global_active_blocks: list["ContextBlock"] = (
        []
    )  # track what context blocks are globally active

    _grammar_only = 0  # a flag that tracks when we are forced to be executing only compiled grammars (like when we are inside a select)

    def __init__(self, engine, echo=True, parent_id=None, **kwargs):
        """Build a new model object that represents a model in a given state.

        Note that this constructor is not meant to be used directly, since there

        Parameters
        ----------
        engine : Engine
            The inference engine to use for this model.
        echo : bool
            If true the final result of creating this model state will be displayed (as HTML in a notebook).
        parent_id : int
            Parent model's identifier.
        """
        if isinstance(engine, str) and engine.startswith("http"):
            from ._remote import RemoteEngine

            engine = RemoteEngine(engine, **kwargs)

        # # auto-wrap the tokenizer in the standard guidance interface
        # if not isinstance(tokenizer, Tokenizer):
        #     tokenizer = Tokenizer(tokenizer)

        self.engine = engine
        self.chat_template = (
            engine.get_chat_template()
        )  # TODO [HN]: Should this be a method or attr?
        # NOTE(nopdive): `echo` seems to be better on the engine, when is there an opportunity to turn echo off midway?
        self.echo = echo
        self.token_count = 0  # tracks how many tokens our byte state represents
        self.max_display_rate = (
            0.2  # this controls how frequently we are allowed to redraw the display (in seconds)
        )
        self.opened_blocks = {}  # what context blocks have been opened but not closed
        # self.compute_log_probs = compute_log_probs

        # private attributes
        self._variables = {}  # these are the state variables stored with the model
        self._variables_log_probs = {}  # these are the state variables stored with the model
        self._variables_positions = {}  # these are the state variables stored with the model
        self._cache_state = {}  # mutable caching state used to save computation
        self._state = ""  # the current bytes that represent the state of the model
        self._trace_nodes = set()  # keep trace node reference for pinning
        self._event_queue = (
            None  # TODO: these are for streaming results in code, but that needs implemented
        )
        self._event_parent = None
        self._last_display = 0  # used to track the last display call to enable throttling
        self._last_event_stream = (
            0  # used to track the last event streaming call to enable throttling
        )
        self._last_event_stream = (
            0  # used to track the last event streaming call to enable throttling
        )

        self._id = self.__class__.gen_id()  # model id needed for tracking state
        self._parent_id = parent_id
        self._parent: "Model" = None
        self._update_trace_node(self._id, self._parent_id, None)

        self.vis_chunk: VisBytesChunk = None
        self.engine.model_dict[self._id] = self
        self.metrics = GuidanceEngineMetrics()

        weakref.finalize(self, log_cleanup, f"model({id(self)})")
        log_init(f"model({id(self)})")

    @property
    def renderer(self):
        if self.echo:
            return self.engine.renderer
        else:
            return None

    @classmethod
    def gen_id(cls):
        global _id_counter

        _id = _id_counter
        _id_counter += 1
        return _id

    @property
    def active_role_end(self):
        """The default end patterns we should use for `gen` calls.
        TODO: move this logic into the gen call...we can do with if we allow model_variables to run functions.

        These patterns are computed dynamically by the model object because they can depend on
        what the current open roles are, which is something
        """

        # add any active non-empty role ends. Ignore role ends that are spaces
        parts = []
        for _, role_end_str in self.opened_blocks.values():
            role_end_str = format_pattern.sub("", role_end_str)
            if len(role_end_str) > 0 and not re.fullmatch(r"\s+", role_end_str):
                parts.append(role_end_str)

        return select(parts)

    def html(self):
        """Displays model as HTML."""
        # NOTE(nopdive): Have this public for now until all widget related issues are sorted out.
        clear_output(wait=True)
        display(HTML(self._html()))

    def _html(self) -> str:
        """Returns HTML string that displays the model object."""

        return trace_node_to_html(
            self.engine.trace_handler.id_node_map[self._id], hasattr(self, "indent_roles")
        )

    def _send_to_event_queue(self, value):
        """For streaming in code.

        TODO: Is this still needed?"""
        if self._event_queue is not None:
            self._event_queue.put(value)
        if self._event_parent is not None:
            self._event_parent._send_to_event_queue(value)

    def stream(self):
        return ModelStream(self)

    def copy(self):
        """Create a shallow copy of the model object."""

        # start with a shallow copy
        new_lm = copy.copy(self)

        # then copy a few things we need deeper copies of
        new_lm._variables = self._variables.copy()
        new_lm._variables_log_probs = self._variables_log_probs.copy()
        new_lm._variables_positions = self._variables_positions.copy()
        new_lm.opened_blocks = self.opened_blocks.copy()

        # create a new clean event queue
        new_lm._event_queue = (
            None  # we start with no event queue because nobody is listening to us yet
        )

        if self._event_queue is not None:
            # if the current lm has an event queue, we make it our parent
            new_lm._event_parent = self

        elif self._event_parent is not None:
            # otherwise if the current event que has an event parent then that is also our parent
            new_lm._event_parent = self._event_parent

        new_lm._id = self.__class__.gen_id()
        new_lm._parent_id = self._id
        new_lm._trace_nodes = set()
        new_lm._update_trace_node(new_lm._id, new_lm._parent_id, None)
        self.engine.model_dict[new_lm._id] = new_lm
        new_lm.vis_chunk = None
        new_lm._parent = self
        new_lm.metrics = self.metrics.model_copy(deep=True)

        weakref.finalize(new_lm, log_cleanup, f"model({id(self)})")
        log_copy(f"model({id(self)})")
        return new_lm

    def _inplace_append(self, value, force_silent=False):
        """This is the base way to add content to the current LM object that is being constructed.

        All updates to the model state should eventually use this function.
        Note this should only be used after making a copy, otherwise immutability would be violated.

        Parameters
        ----------
        value : bytes | str
            The bytes we should append to our current state.
        """

        # update the byte state
        v = value
        if not isinstance(v, str):
            v = str(value)
        self._state += v

        # this is for programmatic streaming among other things
        self._send_to_event_queue(self)

    def reset(self, clear_variables=True):
        """This resets the state of the model object.

        Parameters
        ----------
        clear_variables : bool
            If we should clear all the model object's variables in addition to reseting the byte state.
        """
        # TODO(nopdive): This violates the immutability assumption on model class for users. Remove on confirmation.

        self._state = self._state[:0]
        self.opened_blocks = {}
        if clear_variables:
            self._variables = {}
            self._variables_log_probs = {}
            self._variables_positions = {}
        return self

    def role_opener(self, role_name, **kwargs):
        # TODO [HN]: Temporary change while I instrument chat_template in transformers only.
        # Eventually have all models use chat_template.
        if hasattr(self, "get_role_start"):
            return self.get_role_start(role_name, **kwargs)
        elif hasattr(self, "chat_template"):
            return self.chat_template.get_role_start(role_name)
        else:
            raise Exception(
                f"You need to use a chat model in order the use role blocks like `with {role_name}():`! Perhaps you meant to use the {type(self).__name__}Chat class?"
            )

    def role_closer(self, role_name, **kwargs):
        # TODO [HN]: Temporary change while I instrument chat_template in transformers only.
        # Eventually have all models use chat_template.
        if hasattr(self, "get_role_end"):
            return self.get_role_end(role_name, **kwargs)
        elif hasattr(self, "chat_template"):
            return self.chat_template.get_role_end(role_name)
        else:
            raise Exception(
                f"You need to use a chat model in order the use role blocks like `with {role_name}():`! Perhaps you meant to use the {type(self).__name__}Chat class?"
            )

    def _repr_html_(self):
        if ipython_is_imported:
            clear_output(wait=True)
        return self._html()

    def _current_prompt(self):
        """The current prompt in bytes (which is the state without the context close tags)."""
        return trace_node_to_str(self.engine.trace_handler.id_node_map[self._id])

    def _update_trace_node(
        self, identifier: int, parent_id: Optional[int], node_attr: Optional[NodeAttr]
    ):
        """Updates trace node that corresponds to this model."""

        trace_node = self.engine.trace_handler.update_node(identifier, parent_id, node_attr)
        self._trace_nodes.add(trace_node)

        if self.renderer is not None:
            self.renderer.update(
                TraceMessage(
                    trace_id=identifier,
                    parent_trace_id=parent_id,
                    node_attr=node_attr,
                )
            )

    def __str__(self):
        """A string representation of the current model object (that includes context closers)."""

        # TODO(nopdive): Ensure context closers or no?
        return trace_node_to_str(self.engine.trace_handler.id_node_map[self._id])

    def __add__(self, value):
        """Adding is the primary mechanism for extending model state.

        Parameters
        ----------
        value : guidance grammar
            The grammar used to extend the current model.
        """

        # create the new lm object we will return
        # (we need to do this since Model objects are immutable)
        lm = self.copy()

        # find blocks that are now active, but haven't been opened by lm yet
        enter_blocks = []
        for context in Model.global_active_blocks:
            if context not in lm.opened_blocks:
                enter_blocks.append(context)
                lm.opened_blocks[context] = (0, "")

        # find opened blocks by lm, but are no longer active
        exit_blocks = []
        for context in list(reversed(lm.opened_blocks.keys())):
            if context not in Model.global_active_blocks:
                exit_blocks.append(context)

        # finish any exiting blocks
        for context in exit_blocks:
            pos, close_text = lm.opened_blocks[context]
            del lm.opened_blocks[context]

            # handle variables
            if context.name is not None:
                # TODO(nopdive): Replace with trace traversal.
                v = format_pattern.sub("", lm._state[pos:])
                lm._variables[context.name] = v
                lm._variables_positions[context.name] = pos
                lm._update_trace_node(
                    lm._id, lm._parent_id, CaptureOutput(name=context.name, value=v)
                )

            # add closer
            # TODO(nopdive): Consider removing context closer/opener on confirmation.
            closer_text = context.closer if isinstance(context.closer, str) else ""
            lm._update_trace_node(
                lm._id, lm._parent_id, RoleCloserInput(name=context.name, text=closer_text)
            )
            lm += context.closer
            lm = lm.copy()

        # start any entering blocks
        for context in enter_blocks:
            # add opener
            opener_text = context.opener if isinstance(context.opener, str) else ""
            closer_text = context.closer if isinstance(context.closer, str) else ""

            lm._update_trace_node(
                lm._id,
                lm._parent_id,
                RoleOpenerInput(name=context.name, text=opener_text, closer_text=closer_text),
            )
            lm += context.opener
            lm = lm.copy()

            # store closer for state extraction later
            # TODO(nopdive): Replace with trace traversal.
            lm.opened_blocks[context] = (len(lm._state), closer_text)

            # handle variables
            # NOTE(nopdive): No stack for variables, this process removes shadowed variables?
            if context.name is not None:
                if context.name in lm._variables:
                    del lm._variables[context.name]
                    if context.name in lm._variables_log_probs:
                        del lm._variables_log_probs[context.name]
                    if context.name in lm._variables_positions:
                        del lm._variables_positions[context.name]

        if isinstance(value, TextOutput):
            lm._inplace_append(value.value)
            out = lm
            out._update_trace_node(out._id, out._parent_id, value)
        elif isinstance(value, CaptureOutput):
            out = lm
            out._update_trace_node(out._id, out._parent_id, value)
        elif isinstance(value, (AudioOutput, AudioInput, VideoInput, VideoOutput)):
            # TODO(nopdive): This is used for testing, consider removing when fully attached to model execution.
            out = lm
            out._update_trace_node(out._id, out._parent_id, value)
        elif isinstance(value, str):
            # wrap raw string values

            is_id = False
            parts = re.split(_tag_pattern, value)

            # we have no embedded objects
            if len(parts) == 1:
                lm._update_trace_node(lm._id, lm._parent_id, LiteralInput(value=value))

                lm._inplace_append(value)
                out = lm

                # generate VisBytesChunk so we know this chunk is input
                input_tokens = []
                if self.echo:
                    _bytes = value.encode("utf-8")
                    _tokens = out.engine.tokenizer.encode(_bytes)
                    out.vis_chunk = VisBytesChunk(
                        bytes=_bytes,
                        is_input=True,
                        input_tokens=[
                            GenToken(
                                token_id=_token,
                                prob=1.0,
                                text=to_utf8_or_bytes_string(out.engine.tokenizer.decode([_token])),
                                latency_ms=0,
                                is_generated=False,
                                is_force_forwarded=False,
                                is_input=True,
                            )
                            for _token in _tokens
                        ],
                    )
                    input_tokens = out.vis_chunk.input_tokens

                out._update_trace_node(
                    out._id,
                    out._parent_id,
                    TextOutput(value=value, is_input=True, tokens=input_tokens),
                )

            # if we have embedded objects we have to convert the string to a grammar tree
            else:
                lm._update_trace_node(lm._id, lm._parent_id, EmbeddedInput(value=value))

                partial_grammar = _null_grammar
                lm.suffix = ""
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        lm.suffix = parts[i + 1]
                    if is_id:
                        call = _call_pool[part]
                        if isinstance(call, GrammarFunction):
                            partial_grammar += _call_pool[part]
                        else:
                            lm += partial_grammar
                            lm = _call_pool[part](lm)
                            partial_grammar = _null_grammar
                    elif part != "":
                        partial_grammar += string(part)
                    is_id = not is_id

                out = lm + partial_grammar

        # if we find a null value we do nothing
        elif isinstance(value, Null):
            out = lm

        # run stateless functions (grammar nodes)
        elif isinstance(value, GrammarFunction):
            lm._update_trace_node(lm._id, lm._parent_id, StatelessGuidanceInput(value=value))
            out = lm._run_stateless(value)

        # run stateful functions
        else:
            lm._update_trace_node(lm._id, lm._parent_id, StatefulGuidanceInput(value=value))
            out = value(lm)
            if out is None:
                raise Exception(
                    f"A guidance function returned `None`, not a model object! Did you forget to return the new lm at the end of your function?"
                )
            if not isinstance(out, Model):
                raise Exception(
                    f"A guidance function did not return a model object! Did you try to add a function to a model without calling the function? For example `model + guidance_function()` is correct, while `model + guidance_function` will cause this error."
                )

        return out

    # def endswith(self, s):
    #     '''Checks if the current model state ends with the given value.'''
    #     return self._current_prompt().endswith(s)

    def __len__(self):
        """The string length of the current state.

        TODO: This should change to the byte length...
        """
        return len(str(self))

    def __setitem__(self, key, value):
        raise Exception(
            "Model objects are immutable so you can't use __setitem__! Consider using the .set(key, value) method instead to create a new updated model object."
        )

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]

        # look for named blocks that are still open with the given key as their name
        else:
            for context in list(reversed(self.opened_blocks)):
                if context.name == key:
                    return format_pattern.sub("", self._state[self.opened_blocks[context][0] :])

        raise KeyError(f"Model does not contain the variable '{key}'")

    def __contains__(self, item):
        return item in self._variables

    def get(self, key, default=None):
        """Return the value of a variable, or a default value if the variable is not present.

        Parameters
        ----------
        key : str
            The name of the variable.
        default : any
            The value to return if the variable is not current set.
        """
        return self._variables.get(key, default)

    def setattr(self, key, value):
        """Return a new model with the given model attribute set.

        Parameters
        ----------
        key : str
            The name of the attribute to be set.
        value : any
            The value to set the attribute to.
        """
        copy = self.copy()
        setattr(copy, key, value)
        return copy

    def delattr(self, key):
        """Return a new model with the given attribute deleted.

        Parameters
        ----------
        key : str
            The attribute name to remove.
        """
        copy = self.copy()
        delattr(copy, key)
        return copy

    def set(self, key, value):
        """Return a new model with the given variable value set.

        Parameters
        ----------
        key : str
            The name of the variable to be set.
        value : any
            The value to set the variable to.
        """
        copy = self.copy()
        copy._variables[key] = value
        copy._variables_log_probs[key] = 0.0
        return copy

    def remove(self, key):
        """Return a new model with the given variable deleted.

        Parameters
        ----------
        key : str
            The variable name to remove.
        """
        if key in self._variables:
            copy = self.copy()
            del copy._variables[key]
            if key in copy._variables_log_probs:
                del copy._variables_log_probs[key]
        else:
            copy = self
        return copy

    def log_prob(self, key, default=None):
        """Return the log prob of a variable, or a default value if the variable is not present.

        Parameters
        ----------
        key : str
            The name of the variable.
        default : any
            The value to return if the variable is not current set.
        """
        # TODO: support calling without a key to get the log prob of the whole model
        return self._variables_log_probs.get(key, default)

    # def get_cache(self):
    #     return self.engine.cache

    #     def tool_def(self, functions):

    #         self += """
    # # Tools

    # """
    #         if len(functions) > 0:
    #             self += '''## functions

    # namespace functions {

    # '''
    #         for function in functions:
    #             self += f"""// {function['description']}
    # type {function['name']} = (_: {{"""
    #             for prop_name,prop_data in function["parameters"]["properties"].items():
    #                 if "description" in prop_data:
    #                     self += f"\n// {prop_data['description']}\n"
    #                 self += prop_name
    #                 if prop_name not in function["parameters"]["required"]:
    #                     self += "?"
    #                 self += ": "
    #                 if "enum" in prop_data:
    #                     for enum in prop_data["enum"]:
    #                         self += f'"{enum}"'
    #                         if enum != prop_data["enum"][-1]:
    #                             self += " | "
    #                 else:
    #                     self += prop_data["type"]

    #                 if prop_name != list(function["parameters"]["properties"].keys())[-1]:
    #                     self += ",\n"
    #             self += """
    # }) => any;

    # """
    #             self[function['name']] = function
    #         self += "} // namespace functions\n"

    #         return self

    def _run_stateless(self, stateless_function, temperature=0.0, top_p=1.0, n=1):
        assert (
            Model._grammar_only == 0
        ), "We can't run grammar parsing while in context free mode! (for example inside a block closer)"

        logger.debug("start Model._run_stateless")

        # This needs to be here for streaming
        # if name is not None:
        #     self[name] = ""

        # replace ModelVariables with their actual values (note we save what we replaced so we can restore it later)
        replacements = replace_model_variables(stateless_function, self)

        # start the generation stream
        gen_obj = self.engine(self._current_prompt(), stateless_function, echo=self.echo)

        # we will return a new extended version of ourselves, which we track as `lm`
        lm = self
        lm.engine.metrics.engine_backtrack_tokens = lm.metrics.engine_backtrack_tokens
        lm.engine.metrics.engine_input_tokens = lm.metrics.engine_input_tokens
        lm.engine.metrics.engine_output_tokens = lm.metrics.engine_output_tokens

        # single generation
        if n == 1:
            generated_value = ""
            # logprobs_out = []

            delayed_bytes = b""
            # last_is_generated = False

            for chunk in gen_obj:

                # we make everything full probability if we are not computing uncertainty
                # if not self.engine.compute_log_probs:
                #     chunk.new_bytes_prob = 1.0

                # convert the bytes to a string (delaying if we don't yet have a valid unicode string)
                lm.token_count += chunk.new_token_count
                chunk.new_bytes = delayed_bytes + chunk.new_bytes
                try:
                    new_text = chunk.new_bytes.decode("utf8")
                except UnicodeDecodeError:
                    delayed_bytes = chunk.new_bytes
                    continue
                delayed_bytes = b""

                if chunk.backtrack:
                    lm.engine.metrics.engine_backtrack_tokens += chunk.backtrack

                if len(chunk.new_bytes) > 0:
                    generated_value += new_text

                    # split chunk into generated and force_forwarded parts for better animated visualization
                    if chunk.generated_bytes:
                        lm += TextOutput(
                            value=chunk.generated_bytes.decode("utf8"),
                            is_generated=True,
                            token_count=0,
                            prob=0.0,
                            tokens=chunk.generated_tokens,
                        )

                    if chunk.force_forwarded_bytes:
                        lm += TextOutput(
                            value=chunk.force_forwarded_bytes.decode("utf8"),
                            is_force_forwarded=True,
                            token_count=0,
                            prob=0.0,
                            tokens=chunk.force_forwarded_tokens,
                        )

                if self.echo:
                    lm.vis_chunk = VisBytesChunk(
                        bytes=chunk.new_bytes,
                        is_input=False,
                        # generated_bytes=chunk.generated_bytes,
                        generated_tokens=chunk.generated_tokens,
                        force_forwarded_tokens=chunk.force_forwarded_tokens,
                        backtrack=chunk.backtrack,
                        engine_outputs=chunk.engine_outputs,
                    )

                # last_is_generated = chunk.is_generated
                if len(chunk.capture_groups) > 0:
                    for k in chunk.capture_groups:
                        v = chunk.capture_groups[k]

                        # see if we are in a list_append mode
                        if isinstance(v, list):
                            for i, inner_v in enumerate(v):
                                # convert to a string if possible
                                # TODO: will need to not just always do this once we support images etc.
                                try:
                                    inner_v = (
                                        inner_v.decode("utf8")
                                        if isinstance(inner_v, bytes)
                                        else inner_v
                                    )
                                except UnicodeDecodeError:
                                    pass

                                if k not in lm or not isinstance(lm._variables[k], list):
                                    lm._variables[k] = []
                                    lm += CaptureOutput(name=k)
                                if k not in lm._variables_log_probs or not isinstance(
                                    lm._variables_log_probs[k], list
                                ):
                                    lm._variables_log_probs[k] = []
                                if k not in lm._variables_positions:
                                    lm._variables_positions[k] = len(lm._state)

                                lm._variables[k].append(inner_v)
                                lm._variables_log_probs[k].append(
                                    chunk.capture_group_log_probs[k][i]
                                )
                                lm += CaptureOutput(
                                    name=k,
                                    value=inner_v,
                                    is_append=True,
                                    log_probs=lm._variables_log_probs[k][i],
                                )

                        # ...or standard assignment mode
                        else:
                            # convert to a string if possible
                            # TODO: will need to not just always do this once we support images etc.
                            try:
                                v = v.decode("utf8") if isinstance(v, bytes) else v
                            except UnicodeDecodeError:
                                pass

                            lm._variables[k] = v
                            lm._variables_positions[k] = len(lm._state) - len(v)
                            lm._variables_log_probs[k] = chunk.capture_group_log_probs[k]
                            lm += CaptureOutput(
                                name=k,
                                value=v,
                                log_probs=chunk.capture_group_log_probs[k],
                            )

            # if len(chunk.capture_groups) > 0:
            #     for k in chunk.capture_groups:
            #         v = chunk.capture_groups[k]
            #         lm[k] = v.decode("utf8") if isinstance(v, bytes) else v

        unreplace_model_variables(replacements)

        logger.debug("finish Model._run_stateless")

        lm.metrics = lm.engine.metrics.model_copy(deep=True)

        return lm

    def get_per_token_stats(self) -> list[GenTokenExtra]:
        """Get the per token stats (prob, top-k, latency, etc.) for the model.

        Returns
            List[GenTokenExtra]: a list of GenToken objects.
        """

        paths = []
        model = self
        while model is not None:
            paths.append(model)
            if model._parent_id is None:
                break
            model: "Model" = self.engine.model_dict[model._parent_id]
        paths.reverse()

        vis_chunks: list[VisBytesChunk] = [
            path.vis_chunk for path in paths if path.vis_chunk is not None
        ]

        gen_tokens_infos: list[tuple[int, float, list[GenToken]]] = []
        gen_tokens_indices: list[int] = []
        for vis_chunk in vis_chunks:
            for engine_output in vis_chunk.engine_outputs:
                gen_tokens_infos.append(
                    (
                        engine_output.issued_token.token_id,
                        engine_output.issued_token.latency_ms,
                        engine_output.masked_top_k,
                    )
                )
            gen_tokens_indices.append(len(gen_tokens_infos) - 1)

        text = self._state
        token_ids = self.engine.tokenizer.encode(text.encode("utf-8"))

        # verify if text == encode(decode(text))
        # decoded_text = self.engine.tokenizer.decode(token_ids).decode("utf-8")
        decoded_text = "".join(self.engine.tokenizer.decode([_token]).decode("utf-8") for _token in token_ids)

        token_texts: list[str] = []
        if text == decoded_text:
            for idx, token_id in enumerate(token_ids):
                token_texts.append(self.engine.tokenizer.decode([token_id]).decode("utf-8"))
        else:
            # Some models like phi-3 removes spaces and line breaks
            # Scan through the text and find the missing chunks so we can map the token back into generated chunks by the parser later on
            ptr_idx = 0
            for _idx, token_id in enumerate(token_ids):
                missing_chunk = ""
                decoded = self.engine.tokenizer.decode([token_id]).decode("utf-8")
                if not text[ptr_idx:].startswith(decoded):
                    _next_idx = ptr_idx
                    found = False
                    while _next_idx < len(text):
                        if text[_next_idx:].startswith(decoded):
                            found = True
                            break
                        _next_idx += 1

                    assert found, f"Failed to find the token {decoded} in the text {text}"
                    missing_chunk = text[ptr_idx:_next_idx]

                token_texts.append(missing_chunk + decoded)
                ptr_idx += len(missing_chunk + decoded)

            # verify
            _decoded_text = "".join(token_texts)
            assert _decoded_text in text, f"Failed to decode the text {text} into {_decoded_text}"
            if _decoded_text in text and len(_decoded_text) < len(text):
                # there are some missing chunks (probably empty spaces/ end lines removed by the tokenizer)
                missing_chunk = text[len(_decoded_text):]
                token_texts[-1] += missing_chunk

        # NOTE (loc): Not all engines support the get_logits method
        try:
            # make sure token_ids start with BOS token if needed
            prev_token_ids_len = len(token_ids)
            if (
                len(token_ids) > 0
                and self.engine.tokenizer.bos_token_id != None
                and token_ids[0] != self.engine.tokenizer.bos_token_id
            ):
                token_ids = [self.engine.tokenizer.bos_token_id] + token_ids

            tokens_with_topk = self.engine.get_per_token_topk_probs(token_ids, top_k=self.engine._top_k)

            # remove the BOS token
            if prev_token_ids_len != len(token_ids):
                token_ids = token_ids[1:]
                tokens_with_topk = tokens_with_topk[1:]

        except Exception as e:
            logger.warning(
                f"Failed to get the top k probabilities for the tokens {token_ids}. Error: {e}"
            )
            # FIXME (loc): assume prob 1.0 for all tokens
            tokens_with_topk = []
            for token_id, token_text in zip(token_ids, token_texts):
                tokens_with_topk.append(
                    GenTokenExtra(
                        token_id=token_id,
                        prob=1.0,
                        text=token_text,
                        top_k=[
                            GenToken(token_id=token_id, prob=1.0, text=token_text),
                        ],
                    )
                )

        start_idx = 0
        end_idx = 1
        start_pos = 0
        remainder = ""

        processed_gen_tokens: list[GenTokenExtra] = []
        # Map token back to generated chunk to extract correct info (is_generated, latency, etc.)
        for vis_chunk_idx, vis_chunk in enumerate(vis_chunks):
            vis_text = vis_chunk.bytes.decode("utf-8")

            if not vis_text:
                continue

            # Find the chunk starting at start_idx that contains the vis_text
            end_idx = start_idx
            _chunk = "".join(token_texts[start_idx : end_idx + 1])
            while vis_text not in _chunk and end_idx < len(token_texts):
                # expand the chunk
                end_idx += 1
                _chunk = "".join(token_texts[start_idx : end_idx + 1])

            if vis_text not in _chunk and end_idx >= len(token_texts):
                # failed = True
                # break
                raise Exception(f"Failed to find the {vis_text} in the tokens chunk {_chunk}")

            if vis_text == _chunk:
                # perfect match
                pass
            else:
                start_pos = _chunk.index(vis_text)
                remainder = _chunk[start_pos + len(vis_text) :]

            if remainder:
                # we have a current chunk that is larger than the vis_text
                # probably the last token is a partial token that spans across two chunks due to backtracking
                # we should not issue that token for now
                end_idx -= 1

            _chunk_token_ids = token_ids[start_idx : end_idx + 1]
            _chunk_tokens_with_topk = tokens_with_topk[start_idx : end_idx + 1]
            _chunk_token_texts = token_texts[start_idx : end_idx + 1]

            is_input = len(vis_chunk.input_tokens) > 0
            is_force_forwarded = len(vis_chunk.force_forwarded_tokens) > 0

            _gen_tokens: list[GenTokenExtra] = []
            for token_id, token_info_with_topk, token_text in zip(
                _chunk_token_ids, _chunk_tokens_with_topk, _chunk_token_texts
            ):
                prob = -1
                for _token in token_info_with_topk.top_k:
                    if _token.token_id == token_id:
                        prob = _token.prob
                        break

                _gen_token = GenTokenExtra(
                    token_id=token_id,
                    prob=prob,
                    text=token_text,
                    latency_ms=0,
                    is_input=is_input,
                    is_generated=False,
                    is_force_forwarded=False,
                )
                _gen_token.top_k = token_info_with_topk.top_k
                _gen_tokens.append(_gen_token)

            for i, _gen_token in enumerate(_gen_tokens):
                if not is_input:
                    if i < len(vis_chunk.generated_tokens):
                        _gen_token.is_generated = True
                    else:
                        if is_force_forwarded:
                            _gen_token.is_force_forwarded = True

                    # Start from the end of current chunk
                    # go backwards to find the match between token and associated text string
                    found_perfect_match = False
                    max_idx = gen_tokens_indices[vis_chunk_idx]
                    for idx in range(max_idx, -1, -1):
                        if _gen_token.token_id == gen_tokens_infos[idx][0]:
                            _gen_token.latency_ms = gen_tokens_infos[idx][1]
                            _masked_top_k = gen_tokens_infos[idx][2]

                            # if we find a match in tokens list generated by the engine, then this token should be marked as generated
                            _gen_token.is_generated = True
                            _gen_token.is_force_forwarded = False

                            if _masked_top_k is None:
                                # in free accepting state, no masking
                                for _token in _gen_token.top_k:
                                    _token.is_masked = False
                            else:
                                _masked_tokens = [token.token_id for token in _masked_top_k]
                                for _token in _gen_token.top_k:
                                    if _token.token_id not in _masked_tokens:
                                        _token.is_masked = True
                                    else:
                                        _token.is_masked = False

                            found_perfect_match = True
                            break

                    # NOTE (loc): There are cases that the generated token and issued token are not matched
                    # for example, the engine may issue token "pl" but the parser decides to generate token "plate" due to the constraints
                    # To mitigate the issue, we narrow down the search space to find the text that may contain the generated token
                    if not found_perfect_match:
                        # only search within this chunk
                        max_idx = gen_tokens_indices[vis_chunk_idx]
                        prev_max_idx = (
                            -1 if vis_chunk_idx == 0 else gen_tokens_indices[vis_chunk_idx - 1] - 1
                        )
                        for idx in range(max_idx, prev_max_idx, -1):
                            if (
                                self.engine.tokenizer.decode([gen_tokens_infos[idx][0]]).decode(
                                    "utf-8"
                                )
                                in _gen_token.text
                            ):
                                _gen_token.latency_ms = gen_tokens_infos[idx][1]
                                _masked_top_k = gen_tokens_infos[idx][2]

                                # if we find a match in tokens list generated by the engine, then this token should be marked as generated
                                _gen_token.is_generated = True
                                _gen_token.is_force_forwarded = False

                                if _masked_top_k is None:
                                    # in free accepting state, no masking
                                    for _token in _gen_token.top_k:
                                        _token.is_masked = False
                                else:
                                    _masked_tokens = [token.token_id for token in _masked_top_k]
                                    for _token in _gen_token.top_k:
                                        if (
                                            _token.token_id not in _masked_tokens
                                            and _token.token_id != _gen_token.token_id
                                        ):
                                            _token.is_masked = True
                                        else:
                                            _token.is_masked = False

                                break
                else:
                    # input tokens are not masked
                    for _token in _gen_token.top_k:
                        _token.is_masked = False

            for _gen_token in _gen_tokens:
                if _gen_token.text != "":
                    processed_gen_tokens.append(_gen_token)

            start_idx = end_idx + 1

            start_pos = 0
            remainder = ""

        if not self._variables:
            return processed_gen_tokens
        
        def find_start_and_end_positions(chunk: str, start_idx: int, end_idx: int) -> tuple[int, int]:
            s = "".join(gen_token.text for gen_token in processed_gen_tokens[start_idx:end_idx+1])
            if chunk not in s:
                # chunk is not in the string
                return (-1, -1)
            
            i = start_idx + (end_idx - start_idx) // 2
            traversed = []
            while True:
                if i in traversed:
                    break

                s = "".join(gen_token.text for gen_token in processed_gen_tokens[i:end_idx+1])
                traversed.append(i)

                if chunk in s:
                    start_idx = i
                    # shift the start position to the right
                    i += (end_idx - i) // 2
                else:
                    # shift the start position to the left
                    i -= (i - start_idx) // 2

            j = end_idx
            traversed = []
            while True:
                if j in traversed:
                    break

                s = "".join(gen_token.text for gen_token in processed_gen_tokens[start_idx:j+1])
                traversed.append(j)

                if chunk in s:
                    end_idx = j
                    # shift the end position to the left
                    j -= (j - start_idx) // 2
                else:
                    # shift the end position to the right
                    j += (end_idx - j) // 2

            return (start_idx, end_idx)

        if self.engine.compute_log_probs:
            for k in self._variables:
                try:
                    chunk = self._variables[k]
                    chunk_start_pos = self._variables_positions[k]
                    # find the start of this capture in case we have multiple similar captures
                    _, start_idx = find_start_and_end_positions(self._state[:chunk_start_pos+1], 0, len(processed_gen_tokens)-1)
                    start_idx, end_idx = find_start_and_end_positions(chunk, start_idx, len(processed_gen_tokens)-1)
                    if start_idx == -1:
                        continue

                    self._variables_log_probs[k] = [
                        {
                            "token": token.text,
                            "token_id": token.token_id,
                            "logprob": np.log(token.prob).item()
                        }
                        for token in processed_gen_tokens[start_idx:end_idx+1]
                    ]
                except:
                    pass

        return processed_gen_tokens


class ModelStream:
    def __init__(self, model, grammar=None, timeout=5):
        """Create a model stream object that delays execution until it is iterated over."""
        if model.echo:
            model = model.copy()
            model.echo = False  # turn off display echoing
        self.model = model
        self.grammar = grammar
        self.timeout = timeout

    def __add__(self, grammar):
        """Extend this delayed chain of execution with another grammar append."""
        if self.grammar is None:
            return ModelStream(self.model, grammar)
        else:
            return ModelStream(self.model, self.grammar + grammar)

    def _inner_run(self, model):
        """This runs the model stream without iterating, and is only using internally by __iter__."""
        if isinstance(self.grammar, ModelStream):
            model = self.grammar._inner_run(model)
        elif self.grammar is None:
            model = self.model + ""
        else:
            model = self.model + self.grammar

    def __iter__(self):
        """Starts a thread to execute the model and grammar, yielding events as they occur."""

        # Create a thread-safe queue to hold events
        with CaptureEvents(self.model) as events:

            # Define the target function for the thread
            def target():
                try:
                    self._inner_run(self.model)
                    events.put(None)  # mark that we are done
                except BaseException as ex:
                    events.put(ex)

            # Start the thread
            thread = threading.Thread(target=target)
            thread.start()

            # Yield events from the queue as they become available
            while True:
                try:
                    # Wait for an event with a timeout to allow for thread termination
                    event = events.get(timeout=self.timeout)
                    if event is None:
                        break
                    elif isinstance(event, BaseException):
                        raise event
                    yield event
                except queue.Empty:
                    # Check if the thread is still alive
                    if not thread.is_alive():
                        break

            # Ensure the thread has completed
            thread.join()


class Chat(Model):
    """The base class for all chat-tuned models."""

    def get_role_start(self, role_name, **kwargs):
        """The starting grammar for a role.

        By default we follow the GPT role tag start conventions.

        Parameters
        ----------
        role_name : str
            The name of the role, like "user", or "assistant"
        kwargs : dict
            This kwargs are added to the role start as arguments.
        """
        return (
            "<|im_start|>" + role_name + "".join([f' {k}="{v}"' for k, v in kwargs.items()]) + "\n"
        )

    def get_role_end(self, role_name=None):
        """The ending bytes for a role.

        Note that we cannot use a grammar in closers because they need to remain constant
        so we can append them whenever we need a representation before the final closing of the context.
        By default we follow the GPT role tag end conventions.

        Parameters
        ----------
        role_name : str
            The name of the role, like "user", or "assistant"
        """
        return "<|im_end|>"


class Instruct(Model):
    """The base class for all instruction-tuned models."""

    def get_role_start(self, role_name, **kwargs):
        raise Exception("Subclasses need to define what the role start should be!")

    def get_role_end(self, role_name=None):
        raise Exception("Subclasses need to define what the role end should be!")


class GrammarOnly:
    def __enter__(self):
        Model._grammar_only += 1

    def __exit__(self, exc_type, exc_value, traceback):
        Model._grammar_only -= 1


def grammar_only():
    """Returns a context manager that ensures only grammars are executed (not full python functions)."""
    return GrammarOnly()


class ConstraintException(Exception):
    def __init__(self, *args, **kwargs):
        self.prompt = kwargs.pop("prompt", None)
        self.data = kwargs.pop("data", None)
        super().__init__(*args, **kwargs)


class MonitoringMetric(str, Enum):
    CPU_USAGE = "cpu_usage"
    MEM_USAGE = "mem_usage"
    GPU_USAGE = "gpu_usage"
    GPU_USED_MEM = "gpu_used_mem"
    GPU_TOTAL_MEM = "gpu_total_mem"
    INPUT_TOKENS = "input_tokens"
    OUTPUT_TOKENS = "output_tokens"
    BACKTRACK_TOKENS = "backtrack_tokens"
    TOKEN_COUNT = "token_count"
    TOKEN_REDUCTION = "token_reduction"
    AVG_LATENCY = "avg_latency"


ALL_METRICS = [
    MonitoringMetric.CPU_USAGE,
    MonitoringMetric.MEM_USAGE,
    MonitoringMetric.GPU_USAGE,
    MonitoringMetric.GPU_USED_MEM,
    MonitoringMetric.GPU_TOTAL_MEM,
    MonitoringMetric.INPUT_TOKENS,
    MonitoringMetric.OUTPUT_TOKENS,
    MonitoringMetric.BACKTRACK_TOKENS,
    MonitoringMetric.TOKEN_COUNT,
    MonitoringMetric.TOKEN_REDUCTION,
    MonitoringMetric.AVG_LATENCY,
]


def _monitor_fn(
    stop_flag,
    metrics_dict: dict[MonitoringMetric, list],
    max_size: int = 100,
    interval_ms: float = 1000,
):
    # print("Monitoring started")

    to_collect_gpu_stats = False
    has_gpustat = False
    try:
        import gpustat

        has_gpustat = True
    except:
        logger.warning("gpustat is not installed, run `pip install gpustat` to collect GPU stats.")

    if has_gpustat:
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            if len(gpu_stats) > 0:
                # only collect GPU stats if there is at least one GPU
                to_collect_gpu_stats = True
        except:
            logger.warning("Non-Nvidia GPU monitoring is not supported in this version.")

    try:
        while not stop_flag.value:
            t0 = time.time()

            # cpu_percent = psutil.cpu_percent(interval=1)
            cpu_percent = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory()

            metrics_dict[MonitoringMetric.CPU_USAGE].append(cpu_percent)
            metrics_dict[MonitoringMetric.MEM_USAGE].append(memory_usage.percent)

            if to_collect_gpu_stats:
                gpu_stats = gpustat.GPUStatCollection.new_query()

                usage = [gpu.utilization for gpu in gpu_stats.gpus]
                mem_usage = [gpu.memory_used for gpu in gpu_stats.gpus]
                mem_total = [gpu.memory_total for gpu in gpu_stats.gpus]

                metrics_dict[MonitoringMetric.GPU_USAGE].append(usage)
                metrics_dict[MonitoringMetric.GPU_USED_MEM].append(mem_usage)
                metrics_dict[MonitoringMetric.GPU_TOTAL_MEM].append(mem_total)

            for metrics in metrics_dict.values():
                if len(metrics) > max_size:
                    metrics.pop(0)

            lat = time.time() - t0

            # sleep for the remaining time of the interval
            sleep_time = interval_ms / 1000.0 - lat
            if sleep_time < 0:
                time.sleep(sleep_time)
    except Exception as e:
        # print(f"Error in monitoring: {e}")
        pass

    # print("Monitoring stopped")


class Monitor:
    """Monitoring service to collect necessary metrics for visualization"""

    def __init__(self, engine_metrics: GuidanceEngineMetrics, **kwargs):
        self.engine_metrics = engine_metrics
        self.mp_manager = Manager()

        # use list instead of queue for easily accessing each item, e.g., last item
        self.max_size = kwargs.get("max_size", 100)

        self.metrics_dict = {
            MonitoringMetric.CPU_USAGE: self.mp_manager.list(),
            MonitoringMetric.MEM_USAGE: self.mp_manager.list(),
            MonitoringMetric.GPU_USAGE: self.mp_manager.list(),
            MonitoringMetric.GPU_USED_MEM: self.mp_manager.list(),
            MonitoringMetric.GPU_TOTAL_MEM: self.mp_manager.list(),
        }

        self.stop_flag = self.mp_manager.Value("b", False)
        self.process = None

        self.per_token_metrics = []  # store metrics per token in token list

    def start(self):
        self.process = Process(
            target=_monitor_fn, args=(self.stop_flag, self.metrics_dict, self.max_size)
        )
        self.process.start()
        logger.debug("Monitor:start")

    def stop(self):
        if self.process:
            self.stop_flag.value = True
            self.process.terminate()

            for metrics in self.metrics_dict.values():
                metrics[:] = []  # NOTE(nopdive): ListProxy does not have .clear method.
        logger.debug("Monitor:stop")

    def reset(self):
        self.stop()

        for metrics in self.metrics_dict.values():
            metrics.clear()

        self.start()
        logger.debug("Monitor:reset")

    def get_metrics(
        self, metrics=None, lm: Union[Model, None] = None
    ) -> dict[MonitoringMetric, Any]:
        if metrics is None:
            metrics = ALL_METRICS
        result = {}

        for metric in metrics:
            if metric in [
                MonitoringMetric.CPU_USAGE,
                MonitoringMetric.MEM_USAGE,
                MonitoringMetric.GPU_USAGE,
                MonitoringMetric.GPU_USED_MEM,
                MonitoringMetric.GPU_TOTAL_MEM,
            ]:
                result[metric] = (
                    self.metrics_dict[metric][-1] if len(self.metrics_dict[metric]) > 0 else None
                )
            elif metric == MonitoringMetric.INPUT_TOKENS:
                result[metric] = self.engine_metrics.engine_input_tokens
            elif metric == MonitoringMetric.OUTPUT_TOKENS:
                result[metric] = self.engine_metrics.engine_output_tokens
            elif metric == MonitoringMetric.BACKTRACK_TOKENS:
                result[metric] = self.engine_metrics.engine_backtrack_tokens
            elif metric == MonitoringMetric.TOKEN_COUNT:
                result[metric] = lm.token_count if lm is not None else None
            elif metric == MonitoringMetric.TOKEN_REDUCTION:
                if lm is not None and lm.token_count > 0:
                    result[metric] = 1 - min(1, (lm.metrics.engine_output_tokens / lm.token_count))
                else:
                    result[metric] = None
            elif metric == MonitoringMetric.AVG_LATENCY:
                if lm is None:
                    result[metric] = None
                else:
                    lats = []
                    model = lm
                    while model._parent is not None:
                        if model.vis_chunk:
                            for token in model.vis_chunk.generated_tokens:
                                lats.append(token.latency_ms)
                            for token in model.vis_chunk.force_forwarded_tokens:
                                lats.append(token.latency_ms)
                        model = model._parent

                    if len(lats) == 0:
                        result[metric] = None
                    else:
                        result[metric] = np.mean(lats)

        return result

    def get_metric(self, metric: MonitoringMetric, lm: Union[Model, None] = None) -> Any:
        return self.get_metrics([metric], lm)[metric]
