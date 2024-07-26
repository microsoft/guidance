import regex as regex_module
import logging
from .._guidance import guidance
from .._grammar import select, Gen, quote_regex, capture, token_limit, with_temperature
from ._block import block
from ._silent import silent
from ._tool import Tool

logger = logging.getLogger(__name__)


# TODO: make this stateless!
# TODO: uncomment this once we get temperature stateless
@guidance(stateless=lambda *args, **kwargs: kwargs.get("tools", None) is None)
def gen(
    lm,
    name=None,
    *,
    max_tokens=1e10,
    list_append=False,
    regex=None,
    tools=None,
    hide_tool_call=False,
    stop=None,
    stop_regex=None,
    suffix="",
    n=1,
    temperature=0.0,
    top_p=1.0,
    save_stop_text=False,
):
    """Generate a set of tokens until a given stop criteria has been met.

    This function is a useful utility that can allow you to specify most grammars used by typical
    LM generation programs. It also has the added ability to interleave generation with tool calls.

        >>> lm += gen("my_generation", max_tokens=10)
        >>> print(lm["my_generation"])
        some text from the LLM

    Parameters
    ----------

        name : str or None
            If this is not None then the the results of the generation will be saved as a variable on
            the Model object (so you can access the result as `lm["var_name"]`).

        max_tokens : int
            The maximum number of generation tokens we should use. Note that this limit is not exact when
            regular expression pattern constraints are present, but guidance does attempt to end the generation
            as soon as possible while keeping the regex constraints satisfied.

        list_append : bool
            If this is True then the results saved to `lm[name]` will not be written directly but rather appended
            to a list (if no list with the current name is present one will be created). This is useful for
            building lists inside python loops.

        regex : str or None
            This is a regular expression that will be used to constrain the generation. The model is only allowed
            to generate tokens that match this regular expression. Note that for variable length expressions the
            model is free to continue the expression after a complete match, but generation will terminate as soon
            as the model generates anything that does not match the pattern (this ending behavior may change a bit we
            update guidance to maintain the grammar parsing state between calls).

        stop : str or list or None
            The stop string (or list of strings) we should use for terminating this generation segment.

        stop_regex : str or list or None
            The stop regular expression (or list of regular expressions) we should use for terminating this generation segment.

        save_stop_text : bool or str
            If True then this saves the captured stop text or regex into a variable of the name `str(name) + "_stop_text"`. If
            a string is given then the captured stop text is saved under that name.

        temperature : float
            The temperature to use during this generation call. Note that when parsing ambiguous grammars that include
            multiple conflicting temperatures (for example from multiple possible `gen` calls inside a `select`) the highest
            temperature of all options is used by the model (since we only want to run the model once, not once for every
            possible parse path).

        top_p : float
            TODO! Will control the models top_p generation parameter, but has been yet been implemented beyond top_p=1.0.

        n : int
            TODO! Will control the number of parallel generation calls made during gen.

        tools : Tool or list or None
            A list of guidance.Tool or python functions (which will be converted to guidance.Tool)

        hide_tool_call : bool
            Controls if we should hide the text generated by the model to trigger a tool call. You may want to hide the tool
            call from the model's context if you plan to change it's format after the call is made.
    """
    # TODO: expand the tools doc string
    if [tools, regex].count(None) == 0:
            raise ValueError("Cannot use regex with tools")

    assert (
        n == 1
    ), "We still need to add support for n>1! Consider putting your gen call in a loop for now."
    assert top_p == 1, "We still need to add support for top_p != 1!"

    logger.debug(f'start gen(name="{name}")')

    if stop is None and stop_regex is None and suffix != "":
        stop = suffix

    # Empty stop condition is implicitly the EOS token
    gen_stop = ""
    if stop is not False:
        if stop is None:
            stop = []
        if isinstance(stop, str):
            stop = [stop]

        if stop_regex is None:
            stop_regex = []
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]

        stop_regex += [quote_regex(s) for s in stop]
        if len(stop_regex) == 1:
            gen_stop = stop_regex[0]
        else:
            gen_stop = "|".join("(" + s + ")" for s in stop_regex)

    if regex is None:
        regex = r"(?s:.*)"
    if save_stop_text is True:
        save_stop_text = str(name) + "_stop_text"
    if not isinstance(save_stop_text, str):
        save_stop_text = None

    if tools is not None:
        tools = [Tool(callable=x) if not isinstance(x, Tool) else x for x in tools]
        options = []#Gen(body_regex=regex, stop_regex=gen_stop, save_stop_text=save_stop_text, max_tokens=max_tokens)]
        for i, tool in enumerate(tools):
            # Infer a regex that will match the start of a tool call
            tool_call_prefix = tool.call_grammar.forced_prefix()
            if len(tool_call_prefix) < 4:
                # TODO: alternatively check that the prefix contains the name (case insensitive) of the tool?
                # anything shorter is probably far too ambiguous
                raise ValueError(f"Could not infer unambiguous tool call prefix for tool {tool.name}")
            options.append(
                capture(
                    Gen(body_regex=regex, stop_regex=quote_regex(tool_call_prefix), max_tokens=max_tokens),
                    name=f"tool{i}"
                )
            )
        grm = select(options)
        initial_token_count = lm.token_count
        while lm.token_count <= max_tokens + initial_token_count:
            lm += grm
            tool_called = False
            for i in range(len(tools)):
                tool_i = f"tool{i}"
                if tool_i in lm:
                    tool_called = True
                    if hide_tool_call:
                        temp_lm = lm + tools[i].call_grammar
                        with block("tool_call"):
                            temp_lm += tools[i].tool_call()
                        lm += temp_lm["tool_call"]
                    else:
                        lm += tools[i].call_grammar + tools[i].tool_call()
                lm.remove(tool_i)
            if not tool_called:
                lm += suffix
                break
        return lm
                
    pattern = Gen(body_regex=regex, stop_regex=gen_stop, save_stop_text=save_stop_text, max_tokens=max_tokens)

    tagged_name = "__LIST_APPEND:" + name if list_append and name is not None else name

    # define any capture group for non-tool calls
    if name is not None and tools is None:
        pattern = capture(pattern, name=tagged_name)

    # limit the number of tokens
    pattern = token_limit(pattern, max_tokens)
    lm += with_temperature(pattern + suffix, temperature)

    logger.debug(f"finish gen")
    return lm


def click_loop_start(id, total_count, echo, color):
    click_script = (
        """
function cycle_IDVAL(button_el) {
var i = 0;
while (i < 50) {
var el = document.getElementById("IDVAL_" + i);
if (el.style.display == "inline") {
    el.style.display = "none";
    var next_el = document.getElementById("IDVAL_" + (i+1));
    if (!next_el) {
        next_el = document.getElementById("IDVAL_0");
    }
    if (next_el) {
        next_el.style.display = "inline";
    }
    break;
}
i += 1;
}
button_el.innerHTML = (((i+1) % TOTALCOUNT) + 1)  + "/" + TOTALCOUNT;
}
cycle_IDVAL(this);""".replace(
            "IDVAL", id
        )
        .replace("TOTALCOUNT", str(total_count))
        .replace("\n", "")
    )
    out = f"""<div style='background: rgba(255, 255, 255, 0.0); border-radius: 4px 0px 0px 4px; border: 1px solid {color}; border-right: 0px; padding-left: 3px; padding-right: 3px; user-select: none; color: {color}; display: inline; font-weight: normal; cursor: pointer' onClick='{click_script}'>1/{total_count}</div>"""
    out += f"<div style='display: inline;' id='{id}_0'>"
    return "<||_html:" + out + "_||>"


def click_loop_mid(id, index, echo):
    alpha = 1.0 if not echo else 0.5
    out = f"</div><div style='display: none; opacity: {alpha}' id='{id}_{index}'>"
    return "<||_html:" + out + "_||>"


@guidance
def gen_line(lm, *args, **kwargs):
    return lm.gen(*args, suffix="\n", **kwargs)


@guidance
def gen_quote(lm, name=None, quote='"', *args, **kwargs):
    return lm(quote).gen(*args, name=name, suffix=quote, **kwargs)


@guidance
def will_gen(lm, stop=None, stop_regex=None, ignore_spaces=False, max_tokens=30):
    # this is obviously not the right implementation, just here so we can explore
    if stop and not isinstance(stop, list):
        stop = [stop]
    if stop_regex and not isinstance(stop_regex, list):
        stop_regex = [stop_regex]
    assert (stop is not None) or (stop_regex is not None)
    if not stop:
        stop = []
    if not stop_regex:
        stop_regex = []
    regexes = [regex_module.escape(x) for x in stop + stop_regex]
    optional_space = "\\s*" if ignore_spaces else ""
    pattern = regex_module.compile(f'{optional_space}({"|".join(regexes)})')
    lm2 = lm
    with silent():
        for _ in range(max_tokens):
            lm2 += gen("temp_variable", list_append=True, max_tokens=1)
            if not lm2["temp_variable"] or not pattern.match(
                "".join(lm2["temp_variable"]), partial=True
            ):
                return False
            if pattern.match("".join(lm2["temp_variable"]), partial=False):
                return True
    return False


@guidance
def call_tool(lm, tool):
    lm += tool.call_grammar
    lm += tool.tool_call()
    return lm


@guidance(stateless=True)
def regex(lm, pattern, *, name=None):
    return lm + gen(regex=pattern, name=name)
