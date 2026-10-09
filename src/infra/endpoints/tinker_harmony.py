# ABOUTME: Evaluation-independent GPT-OSS rendering and parsing through OpenAI's Harmony library.
# ABOUTME: Preserves raw arguments; never repairs JSON, resamples model output, or adds eval instructions.
import tinker
from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyEncodingName,
    Message, ReasoningEffort, RenderConversationConfig, Role, SystemContent, TextContent, ToolDescription,
    load_harmony_encoding)


class HarmonyRenderer:
    """Small bridge between our chat transport and OpenAI's reference implementation."""
    def __init__(self, reasoning_effort='medium', current_date=None):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort, self.current_date = reasoning_effort, current_date
        self.tokenizer = self
        self._return_token, self._call_token = 200002, 200012

    def encode(self, text, **kwargs):
        return self.encoding.encode(text, allowed_special='all')

    def decode(self, tokens, **kwargs):
        # Display text may end inside a UTF-8 codepoint. Keep exact tokens in traces.
        return self.encoding.decode(tokens, errors='replace')

    def get_stop_sequences(self):
        return self.encoding.stop_tokens_for_assistant_actions()

    def create_conversation_prefix_with_tools(self, tools):
        content = DeveloperContent.new().with_function_tools([
            ToolDescription(name=t['name'], description=t.get('description') or '',
                            parameters=t.get('parameters')) for t in tools])
        return [dict(role='developer', content=content)]


def generation_prompt(renderer, messages):
    system = SystemContent.new().with_reasoning_effort(ReasoningEffort(renderer.reasoning_effort.title()))
    if renderer.current_date:
        system = system.with_conversation_start_date(renderer.current_date)
    conversation = [Message.from_role_and_content(Role.SYSTEM, system)]
    for item in messages:
        role, content = item['role'], item['content']
        if role == 'assistant':
            parts = content if isinstance(content, list) else [dict(type='text', text=content)]
            thinking = ''.join(p['thinking'] for p in parts if p['type'] == 'thinking')
            text = ''.join(p['text'] for p in parts if p['type'] == 'text')
            calls = item.get('tool_calls') or []
            if thinking:
                conversation.append(Message.from_role_and_content(Role.ASSISTANT, thinking).with_channel('analysis'))
            if text or not calls:
                # Malformed replies are not final answers. Ordinary chat replies are.
                boundary = item.get('harmony_boundary', 'final')
                channel = 'commentary' if calls or boundary != 'final' else 'final'
                conversation.append(Message.from_role_and_content(Role.ASSISTANT, text).with_channel(channel))
            for call in calls:
                conversation.append(Message.from_role_and_content(Role.ASSISTANT, call.function.arguments)
                    .with_channel('commentary').with_recipient('functions.' + call.function.name))
        elif role == 'tool':
            conversation.append(Message(author=Author.new(Role.TOOL, 'functions.' + item['name']),
                                        content=[TextContent(text=content)]))
        else:
            if role == 'developer' and isinstance(content, str):
                content = DeveloperContent.new().with_instructions(content)
            conversation.append(Message.from_role_and_content(Role(role), content))
    tokens = renderer.encoding.render_conversation_for_completion(
        Conversation.from_messages(conversation), Role.ASSISTANT,
        RenderConversationConfig(auto_drop_analysis=True))
    return tinker.ModelInput.from_ints(tokens)


def parse_completion(renderer, tokens, stop_reason):
    """Use OpenAI's parser; expose complete calls with unchanged, possibly invalid JSON."""
    stops = renderer.get_stop_sequences()
    boundaries = [(i, t) for i, t in enumerate(tokens) if t in stops]
    boundary_ok = len(boundaries) == 1 and boundaries[0][0] == len(tokens)-1
    try:
        # OpenAI documents permissive parsing for malformed headers. This is
        # parsing the same sample, not repairing its arguments or resampling it.
        parts = renderer.encoding.parse_messages_from_completion_tokens(tokens, Role.ASSISTANT, strict=False)
    except (ValueError, RuntimeError):
        parts = []
    out, calls, reasoning, content = dict(role='assistant', content=''), [], [], []
    malformed = False
    for part in parts:
        text = ''.join(c.text for c in part.content if hasattr(c, 'text'))
        if part.author.role != Role.ASSISTANT:
            malformed = True
        elif part.recipient:
            if not part.recipient.startswith('functions.') or not part.recipient[len('functions.'):]:
                malformed = True
            else:
                calls.append(dict(type='function', function=dict(name=part.recipient[len('functions.'):], arguments=text)))
        elif part.channel == 'analysis':
            reasoning.append(text)
        elif part.channel in ('final', 'commentary'):
            content.append(text)
        else:
            malformed = True
    out['content'] = ''.join(content)
    if reasoning:
        out['reasoning'] = out['reasoning_content'] = ''.join(reasoning)
    if stop_reason == 'length':
        return out, 'length', 'response_limit'
    if not boundary_ok:
        return out, 'stop', 'malformed_boundary'
    if malformed or not parts:
        return out, 'stop', 'malformed_header'
    if calls:
        out['tool_calls'] = calls
        return out, 'tool_calls', 'handoff' if tokens[-1] == renderer._call_token else 'tool_call_return'
    if tokens[-1] == renderer._return_token:
        return out, 'stop', 'final'
    return out, 'stop', 'boundary_recipient_mismatch'
