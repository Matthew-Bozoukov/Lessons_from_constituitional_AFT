# ABOUTME: Build a small HF task-browser index from official grading and immutable final attempts.
# ABOUTME: Locate literal repeated text with exact offsets; diagnoses never change evaluation outcomes.
import hashlib
import json
from pathlib import Path
import re

POLICY = 'literal-paragraph-cycle-v2'


def repeated_span(text):
    """Conservative same-response loop: >=8 exact cycles, >=2k chars, >=50% of text."""
    if len(text) < 2000:
        return None
    paragraphs = list(re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)', text))
    positions = {}
    for p in paragraphs:
        positions.setdefault(p.group(), []).append(p.start())
    best = None
    for starts in positions.values():
        for i, start in enumerate(starts):
            for second in starts[i+1:]:
                size = second-start
                if start+8*size > len(text):
                    break
                unit = text[start:second]
                end = second
                while text.startswith(unit, end):
                    end += size
                repeats = (end-start)//size
                if repeats >= 8 and end-start >= max(2000, len(text)*.5):
                    if best is None or end-start > best['characters']:
                        # Browser offsets use UTF-16 code units, including astral characters.
                        utf16 = lambda value: len(value.encode('utf-16-le'))//2
                        best = dict(start=utf16(text[:start]), unit_length=utf16(unit), repeats=repeats,
                                    characters=end-start, fraction=(end-start)/len(text),
                                    unit_sha256=hashlib.sha256(unit.encode()).hexdigest(),
                                    text_sha256=hashlib.sha256(text.encode()).hexdigest())
    return best


def diagnose(trajectory):
    loops = []
    markers = []
    gaps = 0
    messages = trajectory.get('messages', [])
    for i, message in enumerate(messages):
        if message.get('role') == 'user' and 'Tool call error:' in str(message.get('content', '')):
            # A preserved rejected response can be followed by tool rejection feedback.
            preceding = next((m for m in reversed(messages[:i]) if m.get('role') != 'tool'), {})
            gaps += not preceding.get('extra', {}).get('format_error_preserved', False)
        if message.get('role') != 'assistant':
            continue
        raw = message.get('extra', {}).get('response', {}).get('choices', [])
        response = raw[0].get('message', {}) if raw else message
        if response.get('reasoning') or response.get('reasoning_content'):
            found = [s for s in ('<think>', '</think>', '<|im_start|>', '<|im_end|>')
                     if s in (response.get('content') or '')]
            if found:
                markers.append({'message_index': i, 'markers': found,
                                'interpretation': 'diagnostic only; may be quoted source text'})
        for field in ('reasoning_content', 'reasoning', 'content'):
            text = response.get(field)
            if isinstance(text, str):
                span = repeated_span(text)
                if span:
                    loops.append(dict(message_index=i, field=field, **span))
    return dict(loops=loops, discarded_response_corrections=gaps, marker_diagnostics=markers)


def build_index(root):
    root = Path(root)
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    state = read(root/'metadata/state.json')
    results_path = root/'results/results.json'
    results = read(results_path) if results_path.exists() else {}
    cached_path = root/'metadata/task-browser.json'
    cached = read(cached_path) if cached_path.exists() else {}
    previous = {r['id']: r for r in cached.get('tasks', [])} if cached.get('policy') == POLICY else {}
    tasks = []
    for iid, task in sorted(state['tasks'].items()):
        valid = [a for a in task['attempts'] if a.get('valid')]
        attempt = valid[-1] if valid else None
        grade = results.get('task_results', {}).get(iid, {})
        row = dict(id=iid, status=task['status'], graded=bool(grade.get('graded')),
                   resolved=grade.get('resolved') if grade.get('graded') else None,
                   attempts=len(task['attempts']), loops=[], discarded_response_corrections=0)
        if attempt:
            path = f"rollouts/{iid}/{attempt['id']}/checkpoint.traj.json"
            payload = (root/path).read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            prior = previous.get(iid, {})
            diagnostic = ({k: prior[k] for k in ('loops', 'discarded_response_corrections', 'marker_diagnostics')}
                          if prior.get('trajectory_sha256') == sha else diagnose(json.loads(payload)))
            trajectory = json.loads(payload)
            end = trajectory['messages'][-1].get('extra', {})
            reason = end.get('limit_reason') or attempt.get('exit_status', '')
            row.update(attempt=attempt['id'], trajectory=path, trajectory_sha256=sha,
                       exit_status=attempt.get('exit_status'), reason=reason, **diagnostic)
        tasks.append(row)
    return dict(schema_version=1, policy=POLICY,
                definition='At least 8 literal consecutive copies, at least 2,000 characters and half of one response field. No causal attribution.',
                tasks=tasks)
