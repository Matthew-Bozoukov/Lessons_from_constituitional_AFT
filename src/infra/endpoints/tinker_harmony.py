# ABOUTME: Evaluation-independent Harmony boundaries, raw tool arguments and prompt rendering.
# ABOUTME: Never repairs model text, executes tools, retries sampling, or adds task-specific instructions.
from tinker_cookbook.renderers.base import Renderer


def generation_prompt(renderer, messages):
    """One standard system preamble, mechanically rendered tools, ordered instructions."""
    system = renderer._get_system_message()
    messages = list(messages)
    routing = [m['content'] for m in messages if m['role'] == renderer._INTERNAL_SYSTEM_ROLE]
    messages = [m for m in messages if m['role'] != renderer._INTERNAL_SYSTEM_ROLE]
    if system:
        system = dict(system, content=system['content'] + ('\n\n' + '\n\n'.join(routing) if routing else ''))
        messages.insert(0, system)
    return Renderer.build_generation_prompt(renderer, messages)


def parse_completion(renderer, tokens, stop_reason):
    """Route a complete Harmony handoff even when its arguments need caller validation.

    JSON/schema validation belongs at the tool execution boundary. Returning raw
    arguments lets that boundary give the model the actual error on the next turn.
    Missing/ambiguous handoffs never expose executable calls.
    """
    raw = renderer.tokenizer.decode(tokens)
    stops = renderer.get_stop_sequences()
    boundaries = [(i, t) for i, t in enumerate(tokens) if t in stops]
    if stop_reason == 'length':
        # Keep analysis separate even on a truncated reply. Never expose a
        # partially generated tool call as executable or as a visible answer.
        parts = renderer._parse_harmony_messages(raw)
        out = dict(role='assistant', content=''.join(p['content'] or '' for p in parts
                   if not p['recipient'] and p['channel'] in ('final','commentary')))
        reasoning = ''.join(p['content'] or '' for p in parts if not p['recipient'] and p['channel']=='analysis')
        if reasoning:
            out['reasoning'] = out['reasoning_content'] = reasoning
        return out, 'length', 'response_limit'
    if len(boundaries) != 1 or boundaries[0][0] != len(tokens)-1:
        return dict(role='assistant', content=raw), 'stop', 'malformed_boundary'
    marker = boundaries[0][1]
    body = renderer.tokenizer.decode(tokens[:-1])
    parts = renderer._parse_harmony_messages(body)
    if not parts:
        return dict(role='assistant', content=raw), 'stop', 'malformed_header'
    calls, reasoning, content = [], [], []
    for part in parts:
        recipient, channel, text = part['recipient'], part['channel'], part['content'] or ''
        if recipient:
            if not recipient.startswith('functions.') or not recipient[len('functions.'):]:
                return dict(role='assistant', content=raw), 'stop', 'unknown_recipient'
            calls.append(dict(type='function', function=dict(name=recipient[len('functions.'):], arguments=text)))
        elif channel == 'analysis':
            reasoning.append(text)
        elif channel in ('final', 'commentary'):
            content.append(text)
        else:
            return dict(role='assistant', content=raw), 'stop', 'malformed_channel'
    out = dict(role='assistant', content=''.join(content))
    if reasoning:
        out['reasoning'] = out['reasoning_content'] = ''.join(reasoning)
    # Match the official cookbook parser: a fully delimited recipient/body
    # remains a tool call with either stop token. Preserve the noncanonical
    # terminal marker as a diagnostic, rather than dropping a complete action.
    if calls:
        out['tool_calls'] = calls
        return out, 'tool_calls', 'handoff' if marker == renderer._call_token else 'tool_call_return'
    if marker == renderer._return_token and not calls:
        return out, 'stop', 'final'
    return dict(role='assistant', content=raw), 'stop', 'boundary_recipient_mismatch'
