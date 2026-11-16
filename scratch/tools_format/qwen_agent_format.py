# ABOUTME: Qwen-Agent's own `qwen` function-calling prompt (the ✿FUNCTION✿ format), copied verbatim, and
# ABOUTME: the edit that moves a row's native `tools` list into its system text in that format.
"""The tools-format arm writes the DA rows' tool definitions the way Qwen-Agent's `QwenFnCallPrompt`
does, instead of handing them to the chat template as `tools=` (Qwen3.6's native `# Tools` /
`<tools>` JSON block, which is also what every eval produces through vLLM).

Strings are verbatim from QwenLM/Qwen-Agent @ 31a4d36d123688581a9e9744427272b33ce940e0,
qwen_agent/llm/fncall_prompts/qwen_fncall_prompt.py: FN_CALL_TEMPLATE_INFO_EN + '\\n\\n' +
FN_CALL_TEMPLATE_FMT_EN (the non-parallel English template; `parallel_function_calls` defaults to
False in qwen_agent/llm/function_calling.py), `get_function_description(..., 'en')`, tool
descriptions joined by '\\n\\n', tool names joined by ','.

One deliberate departure: Qwen-Agent APPENDS the block to an existing system message
('\\n\\n' + block). Here it is PREPENDED (block + '\\n\\n' + operator text), because Qwen3.6's native
block sits before the operator's text; keeping the position fixed leaves format as the only change.
"""
import json

QWEN_AGENT_SOURCE = ("QwenLM/Qwen-Agent@31a4d36d123688581a9e9744427272b33ce940e0:"
                     "qwen_agent/llm/fncall_prompts/qwen_fncall_prompt.py")

FN_NAME = "✿FUNCTION✿"
FN_ARGS = "✿ARGS✿"
FN_RESULT = "✿RESULT✿"
FN_EXIT = "✿RETURN✿"

FN_CALL_TEMPLATE_INFO_EN = """# Tools

## You have access to the following tools:

{tool_descs}"""

FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

%s: The tool to use, should be one of [{tool_names}]
%s: The input of the tool
%s: Tool results
%s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
)

FN_CALL_TEMPLATE_EN = FN_CALL_TEMPLATE_INFO_EN + "\n\n" + FN_CALL_TEMPLATE_FMT_EN

TOOL_DESC_EN = "### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}"
ARGS_FORMAT_EN = "Format the arguments as a JSON object."


def function_description(function: dict) -> str:
    """Qwen-Agent's `get_function_description(function, 'en')` for a function with only
    name/description/parameters (no name_for_human, no code_interpreter, no args_format)."""
    name = function["name"]
    assert name != "code_interpreter", "code_interpreter has its own args_format in Qwen-Agent"
    return TOOL_DESC_EN.format(name_for_human=name, name_for_model=name,
                               description_for_model=function["description"],
                               parameters=json.dumps(function["parameters"], ensure_ascii=False),
                               args_format=ARGS_FORMAT_EN).rstrip()


def tool_system(tools: list[dict]) -> str:
    """The whole tool block for a row's OpenAI-style `tools` list ({type: function, function: {...}})."""
    functions = []
    for t in tools:
        assert t.get("type") == "function" and set(t) == {"type", "function"}, f"unexpected tool shape: {list(t)}"
        assert set(t["function"]) == {"name", "description", "parameters"}, list(t["function"])
        functions.append(t["function"])
    tool_descs = "\n\n".join(function_description(f) for f in functions)
    tool_names = ",".join(f["name"] for f in functions)
    return FN_CALL_TEMPLATE_EN.format(tool_descs=tool_descs, tool_names=tool_names)


def move_tools_to_system(row: dict) -> dict:
    """Return a copy of `row` with its `tools` written into the system message (Qwen-Agent format,
    before the operator text) and the `tools` key removed. Nothing else in the row changes."""
    assert row.get("tools"), "row has no tools to move"
    msgs = row["messages"]
    assert msgs[0]["role"] == "system" and isinstance(msgs[0]["content"], str), "needs a string system turn"
    out = {k: v for k, v in row.items() if k != "tools"}
    sys_msg = dict(msgs[0])
    sys_msg["content"] = tool_system(row["tools"]) + "\n\n" + msgs[0]["content"]
    out["messages"] = [sys_msg] + list(msgs[1:])
    return out
