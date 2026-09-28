# ABOUTME: Lossless APIgen-to-structured-tools conversion for the pinned nosynth control.
# ABOUTME: Removes foreign reasoning for separately recorded GPT-OSS backfill; never invents tool outputs.
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def schema_for(type_text):
    """Parse Python-style APIgen types without executing source text."""
    type_text = type_text.strip()
    optional = bool(re.search(r",\s*optional$", type_text, re.I))
    text = re.sub(r",\s*optional$", "", type_text, flags=re.I)
    def convert(node):
        if isinstance(node, ast.Name):
            names = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
                     "list": "array", "List": "array", "set": "array", "Dict": "object", "dict": "object"}
            if node.id not in names:
                raise ValueError(f"Unsupported type {type_text}")
            out = {"type": names[node.id]}
            if node.id == "set":
                out["uniqueItems"] = True
            return out
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            if name in {"List", "list"}:
                return {"type": "array", "items": convert(args[0])}
            if name == "Union":
                return {"anyOf": [convert(a) for a in args]}
            if name == "Tuple":
                items = [convert(a) for a in args]
                return {"type": "array", "prefixItems": items, "minItems": len(items), "maxItems": len(items),
                        "items": items[0] if all(i == items[0] for i in items) else {"anyOf": items}}
            if name == "Callable":
                # APIgen serializes callable arguments as source/expression strings;
                # retain the exact original type in description and the audit trail.
                return {"type": "string", "description": f"Callable expression serialized as text; original type: {type_text}"}
        raise ValueError(f"Unsupported type expression {type_text}")
    result = convert(ast.parse(text, mode="eval").body)
    result["x-source-python-type"] = type_text
    return result, optional


def convert_row(row, index):
    original = copy.deepcopy(row)
    row = copy.deepcopy(row)
    traces = []
    for turn, m in enumerate(row["messages"]):
        if m.get("reasoning_content"):
            traces.append({"row": index, "turn": turn, "source_trace_sha256": digest(m["reasoning_content"])})
        m.pop("reasoning_content", None)
    tools = []
    kind = "ordinary"
    if row["source"] == "apigen_function_calling":
        system = row["messages"][0]
        match = re.search(r"<tools>(.*?)</tools>", system["content"], re.S)
        if not match:
            raise ValueError(f"Row {index}: missing APIgen schemas")
        raw_tools = json.loads(match.group(1))
        for tool in raw_tools:
            if tool.get("type") == "function" and "function" in tool:
                tools.append(tool)
                continue
            props, required = {}, []
            for name, param in tool.get("parameters", {}).items():
                schema, optional = schema_for(param["type"])
                schema["description"] = " ".join(filter(None, [param.get("description"), schema.get("description")]))
                if "default" in param:
                    schema["default"] = param["default"]
                if not optional and "default" not in param:
                    required.append(name)
                props[name] = schema
            tools.append({"type": "function", "function": {"name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {"type": "object", "properties": props, "required": required}}})
        system["content"] = system["content"].split("You have access to the following tools:")[0].rstrip()
        # Existing task instruction is retained; only the XML transport instruction is removed.
        answer = row["messages"][-1]
        found = re.fullmatch(r"\s*<tool_call>(.*?)</tool_call>\s*", answer["content"], re.S)
        if found:
            calls = json.loads(found.group(1))
            known = {t["function"]["name"] for t in tools}
            if not calls:
                raise ValueError("Unqualified empty-list target")
            output = []
            for j, call in enumerate(calls):
                if call["name"] not in known or not isinstance(call["arguments"], dict):
                    raise ValueError(f"Row {index}: invalid call")
                output.append({"id": f"call_r{index}_n{j}", "type": "function", "function": {
                    "name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}})
            answer["content"] = ""
            answer["tool_calls"] = output
            kind = "single_call" if len(calls) == 1 else "multiple_calls"
        else:
            if "<tool_call" in answer["content"] or not answer["content"].strip():
                raise ValueError(f"Row {index}: malformed APIgen answer")
            kind = "no_applicable_tool"
    # Explicit stable Arrow schema. Arbitrary tool parameter names stay JSON text,
    # not thousands of incompatible sparse nested struct fields.
    row["tools"] = json.dumps(tools, ensure_ascii=False)
    for m in row["messages"]:
        m.setdefault("reasoning_content", None)
        m.setdefault("tool_calls", None)
    return row, traces, {"row": index, "source_sha256": digest(original), "converted_sha256": digest(row), "kind": kind}
