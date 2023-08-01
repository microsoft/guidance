import anthropic
import os
import time
import requests
import copy
import time
import asyncio
import types
import collections
import json
import re
import regex
from decouple import config
from anthropic import Client
ANTHROPIC_API_KEY = config('ANTHROPIC_API_KEY')
from ._llm import LLM, LLMSession, SyncSession


class MalformedPromptException(Exception):
    pass


def prompt_to_messages(prompt):
    messages = []
    # assert prompt.endswith("<|im_start|>Assistant\n"), "When calling Claude chat models you must generate only directly inside the assistant role!"

    pattern = r'<\|im_start\|>(\w+)(.*?)(?=<\|im_end\|>|$)'
    matches = re.findall(pattern, prompt, re.DOTALL)

    if not matches:
        return [{'role': 'user', 'content': prompt.strip()}]

    for match in matches:
        role, content = match
        content = content.strip() # should we do this?
        messages.append({'role': role, 'content': content})

    return messages

def add_text_to_chat_mode_generator(chat_mode):
    for resp in chat_mode:
        if "completion" in resp:
            for c in resp['completion']:
                if "content" in c['delta']:
                    c['text'] = c['delta']['content']
                else:
                    break # the role markers are outside the generation in chat mode right now TODO: consider how this changes for uncontrained generation
            else:
                yield resp
        else:
            yield resp

def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, types.GeneratorType):
        return add_text_to_chat_mode_generator(chat_mode)
    else:
        # for c in chat_mode['completion']:
            # c['text'] = c['message']['content']
        return chat_mode

# model that need to use the chat completion API
chat_models = [
    "claude-instant-1",
    "claude-2"
]

class Anthropic(LLM):
    llm_name: str = "claude"

    def __init__(self, model=None, caching=True, max_retries=3, max_calls_per_min=60,
                 api_key=ANTHROPIC_API_KEY, api_type="claude", api_base=None, api_version=None,
                 temperature=0.0, chat_mode="auto", organization=None, rest_call=False,
                 allowed_special_tokens={"<|endoftext|>", "<|endofprompt|>"},
                 token=None, endpoint=None):
        super().__init__()

        # map old param values
        # TODO: add deprecated warnings after some time
        if token is not None:
            if api_key is None:
                api_key = token
        if endpoint is not None:
            if api_base is None:
                api_base = endpoint

        # fill in default model value
        if model is None:
            model = os.environ.get("ANTHROPIC_MODEL", None)
            print(model)
        if model is None:
            try:
                with open(os.path.expanduser('~/.anthropic_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # fill in default deployment_id value
        # if deployment_id is None:
        #     deployment_id = os.environ.get("ANTHROPIC_DEPLOYMENT_ID", None)

        # auto detect chat completion mode
        if chat_mode == "auto":
            if model in chat_models:
                chat_mode = True
            else:
                chat_mode = False

        # fill in default API key value
        if api_key is None: # get from environment variable
            api_key = os.environ.get("ANTHROPIC_API_KEY", getattr(Anthropic, "api_key", None))
        if api_key is not None and not api_key.startswith("sk-") and os.path.exists(os.path.expanduser(api_key)): # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None: # get from default file location
            try:
                with open(os.path.expanduser('~/.anthropic_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass
        if organization is None:
            organization = os.environ.get("ANTHROPIC_ORGANIZATION", None)
        # fill in default endpoint value
        if api_base is None:
            api_base = os.environ.get("ANTHROPIC_API_BASE", None) or os.environ.get("ANTHROPIC_ENDPOINT", None) # ENDPOINT is deprecated

        import tiktoken
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self.chat_mode = chat_mode

        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        # self.deployment_id = deployment_id
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.organization = organization
        self.rest_call = rest_call

        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

    def session(self, asynchronous=False):
        if asynchronous:
            return AnthropicSession(self)
        else:
            return SyncSession(AnthropicSession(self))

    def role_start(self, role):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return "<|im_start|>"+role+"\n"

    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"

    def end_of_text(self):
        return "<|endoftext|>"

    @classmethod
    def stream_then_save(cls, gen, key, stop_regex, n):
        list_out = []
        cached_out = None

        # init stop_regex variables
        if stop_regex is not None:
            if isinstance(stop_regex, str):
                stop_patterns = [regex.compile(stop_regex)]
            else:
                stop_patterns = [regex.compile(pattern) for pattern in stop_regex]

            current_strings = ["" for _ in range(n)]
            # last_out_pos = ["" for _ in range(n)]

        # iterate through the stream
        all_done = False
        for curr_out in gen:

            # if we have a cached output, extend it with the current output
            if cached_out is not None:
                out = merge_stream_chunks(cached_out, curr_out)
            else:
                out = curr_out

            # check if we have stop_regex matches
            found_partial = False
            if stop_regex is not None:

                # keep track of the generated text so far
                for i,choice in enumerate(curr_out['completion']):
                    current_strings[i] += choice['text']

                # check if all of the strings match a stop string (and hence we can stop the batch inference)
                all_done = True
                for i in range(len(current_strings)):
                    found = False
                    for s in stop_patterns:
                        if s.search(current_strings[i]):
                            found = True
                    if not found:
                        all_done = False
                        break

                # find where trim off the stop regex matches if needed (and look for partial matches)
                stop_pos = [1e10 for _ in range(n)]
                stop_text = [None for _ in range(n)]
                for i in range(len(current_strings)):
                    for s in stop_patterns:
                        m = s.search(current_strings[i], partial=True)
                        if m:
                            span = m.span()
                            if span[1] > span[0]:
                                if m.partial: # we might be starting a stop sequence, so we can't emit anything yet
                                    found_partial = True
                                    break
                                else:
                                    stop_text[i] = current_strings[i][span[0]:span[1]]
                                    stop_pos[i] = min(span[0], stop_pos[i])
                    if stop_pos != 1e10:
                        stop_pos[i] = stop_pos[i] - len(current_strings[i]) # convert to relative position from the end

            # if we might be starting a stop sequence, we need to cache the output and continue to wait and see
            if found_partial:
                cached_out = out
                continue

            # if we get here, we are not starting a stop sequence, so we can emit the output
            else:
                cached_out = None

                if stop_regex is not None:
                    print('out: ', out)
                    for i in range(len(out['completion'])):
                        if stop_pos[i] < len(out['completion'][i]['text']):
                            out['choices'][i] = out['completion'][i].to_dict() # because sometimes we might need to set the text to the empty string (and OpenAI's object does not like that)
                            out['choices'][i]['text'] = out['choices'][i]['text'][:stop_pos[i]]
                            out['choices'][i]['stop_text'] = stop_text[i]
                            out['choices'][i]['finish_reason'] = "stop"

                list_out.append(out)
                yield out
                if all_done:
                    gen.close()
                    break

        # if we have a cached output, emit it
        if cached_out is not None:
            list_out.append(cached_out)
            yield out

        cls.cache[key] = list_out

    def _stream_completion(self):
        pass

    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)

    def _library_call(self, **kwargs):
        """ Call the ANTHROPIC API using the python package.

        """

        # set the params of the openai library if we have them
        if self.api_key is not None:
            anthropic.api_key = self.api_key
        if self.organization is not None:
            anthropic.organization = self.organization
        if self.api_type is not None:
            anthropic.api_type = self.api_type
        if self.api_version is not None:
            anthropic.api_version = self.api_version
        if self.api_base is not None:
            anthropic.api_base = self.api_base

        if self.chat_mode:
            # kwargs['messages'] = prompt_to_messages(kwargs['prompt'])
            kwargs['max_tokens_to_sample'] = 500
            del kwargs['max_tokens']
            del kwargs['echo']
            del kwargs['logprobs']
            del kwargs['function_call']
            # del kwargs['messages']
            # \n\nHuman:
            # print(kwargs)
            # TODO: not sure about this
            anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            out = anthropic_client.completions.create(**kwargs)
            out = add_text_to_chat_mode(out)
        else:
            out = anthropic.Completion.create(**kwargs)

        # restore the params of the anthropic library
        # anthropic.api_key = prev_key
        # anthropic.organization = prev_org
        # anthropic.api_type = prev_type
        # anthropic.api_version = prev_version
        # anthropic.api_base = prev_base

        return out

    def _rest_call(self, **kwargs):
        """ Call the ANTHROPIC API using the REST API.
        """

        # Define the request headers
        headers = copy.copy(self._rest_headers)
        if self.api_key is not None:
            headers['Authorization'] = f"Bearer {self.api_key}"

        # Define the request data
        stream = kwargs.get("stream", False)
        data = {
            "prompt": kwargs["prompt"],
            "max_tokens_to_sample": kwargs.get("max_tokens_to_sample", None),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": stream,
            "logprobs": kwargs.get("logprobs", None),
            'stop': kwargs.get("stop", None),
            "echo": kwargs.get("echo", False)
        }
        if self.chat_mode:
            data['messages'] = prompt_to_messages(data['prompt'])
            del data['prompt']
            del data['echo']
            del data['logprobs']

        # Send a POST request and get the response
        # An exception for timeout is raised if the server has not issued a response for 10 seconds
        try:
            response = requests.post(self.endpoint, headers=headers, json=data, stream=stream, timeout=60)
            if response.status_code != 200:
                raise Exception("Response is not 200: " + response.text)
            if stream:
                return self._rest_stream_handler(response)
            else:
                response = response.json()
        except requests.Timeout:
            raise Exception("Request timed out.")
        except requests.ConnectionError:
            raise Exception("Connection error occurred.")

        if self.chat_mode:
            response = add_text_to_chat_mode(response)
        return response

    def _rest_stream_handler(self, response):
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)

    def encode(self, string):
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(string, allowed_special=self.allowed_special_tokens)

    def decode(self, tokens):
        return self._tokenizer.decode(tokens)


def merge_stream_chunks(first_chunk, second_chunk):

    """ This merges two stream responses together.
    """

    out = copy.deepcopy(first_chunk)

    # merge the choices
    for i in range(len(out['completion'])):
        out_choice = out['completion'][i]
        second_choice = second_chunk['completion'][i]
        out_choice['text'] += second_choice['text']
        if 'index' in second_choice:
            out_choice['index'] = second_choice['index']
        if 'finish_reason' in second_choice:
            out_choice['finish_reason'] = second_choice['finish_reason']
        if out_choice.get('logprobs', None) is not None:
            out_choice['logprobs']['token_logprobs'] += second_choice['logprobs']['token_logprobs']
            out_choice['logprobs']['top_logprobs'] += second_choice['logprobs']['top_logprobs']
            out_choice['logprobs']['text_offset'] = second_choice['logprobs']['text_offset']

    return out


class AnthropicStreamer():
    def __init__(self, stop_regex, n):
        self.stop_regex = stop_regex
        self.n = n
        self.current_strings = ["" for _ in range(n)]
        self.current_length = 0

class RegexStopChecker():
    def __init__(self, stop_pattern, decode, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.decode = decode
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):

        # extend our current strings
        if self.current_strings is None:
            self.current_strings = ["" for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i] += self.decode(input_ids[i][self.current_length:])

        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            for i in range(len(self.current_strings)):
                self.current_strings[i] = self.current_strings[i][self.prefix_length:]

        self.current_length = len(input_ids[0])

        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(self.current_strings[i]):
                    found = True
            if not found:
                all_done = False
                break

        return all_done

# Define a deque to store the timestamps of the calls
class AnthropicSession(LLMSession):
    async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens_to_sample=1000, logprobs=None,
                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
                       cache_seed=0, caching=None, **completion_kwargs):
        """ Generate a completion of the given prompt.
        """

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None
        assert stop_regex is None or stream, "We can only support stop_regex for the Anthropic API when stream=True!"
        assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the Anthropic API!"

        assert token_healing is None or token_healing is False, "The Anthropic API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert not pattern, "The Anthropic API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
        # assert not stop_regex, "The OpenAI API does not support Guidance stop_regex controls! Please either switch to an endpoint that does, or don't use the `stop_regex` argument to `gen`."

        # define the key for the cache
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in llm_cache and stream:
            cache_params["stream"] = False
            key1 = llm_cache.create_key(self.llm.llm_name, **cache_params)
            if key1 in llm_cache:
                key = key1

        # check the cache
        if key not in llm_cache or caching is False or (caching is not True and not self.llm.caching):

            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            fail_count = 0
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "model": self.llm.model_name,
                        # "deployment_id": self.llm.deployment_id,
                        "prompt": prompt,
                        "max_tokens_to_sample": max_tokens_to_sample,
                        "temperature": temperature,
                        "top_p": top_p,
                        "logprobs": logprobs,
                        "echo": echo,
                        "stream": stream,
                        **completion_kwargs
                    }
                    if logit_bias is not None:
                        call_args["logit_bias"] = {str(k): v for k,v in logit_bias.items()} # convert keys to strings since that's the open ai api's format
                    out = self.llm.caller(**call_args)

                except Exception as e:
                    print('error: ', e)
                    await asyncio.sleep(3)
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(f"Too many (more than {self.llm.max_retries}) Anthropic API RateLimitError's in a row!")

            if stream:
                return self.llm.stream_then_save(out, key, stop_regex, n)
            else:
                llm_cache[key] = out

        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]
