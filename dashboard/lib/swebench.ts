// ABOUTME: Revision-pinned SWE-bench task outcomes and lazy original trajectory reads from HF.
// ABOUTME: Grading determines passes; exact original text must verify before any loop is highlighted.
export interface LoopSpan { message_index: number; field: string; start: number; unit_length: number; repeats: number; fraction: number; unit_sha256: string; text_sha256: string }
export interface SweTask { id: string; status: string; graded: boolean; resolved: boolean | null; attempts: number; attempt?: string; trajectory?: string; trajectory_sha256?: string; reason?: string; loops: LoopSpan[]; discarded_response_corrections: number }
export interface SweIndex { schema_version: number; policy: string; definition: string; tasks: SweTask[] }
export interface TrajectoryMessage { role: string; content?: string | null; reasoning_content?: string; reasoning?: string; tool_calls?: unknown[]; extra?: { response?: { choices: { message: Record<string, unknown> }[] } } }
export interface SweTrajectory { messages: TrajectoryMessage[] }
export type Outcome = 'passed' | 'looped' | 'other' | 'ungraded' | 'incomplete';
export function outcome(task: SweTask): Outcome {
  if (task.status !== 'valid') return 'incomplete';
  if (!task.graded || typeof task.resolved !== 'boolean') return 'ungraded';
  if (task.resolved) return 'passed';
  return task.loops.length ? 'looped' : 'other';
}
export function hfUrl(repo: string, revision: string, path: string) {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || path.split('/').includes('..') || !/^[\w.-]+$/.test(revision)) throw new Error('Invalid artifact reference');
  return `https://huggingface.co/datasets/${repo}/resolve/${revision}/${path.split('/').map(encodeURIComponent).join('/')}`;
}
export async function loadSweIndex(repo: string): Promise<{ index: SweIndex; revision: string }> {
  const info = await fetch(`https://huggingface.co/api/datasets/${repo}`);
  if (!info.ok) throw new Error(`Cannot read HF run (${info.status})`);
  const { sha } = await info.json();
  const response = await fetch(hfUrl(repo, sha, 'metadata/task-browser.json'));
  if (response.status === 404) throw new Error('This run has no published task diagnosis index yet. Its raw results remain available in the evaluation explorer. Missing diagnoses are not classified as other failures.');
  if (!response.ok) throw new Error(`Cannot load task diagnoses (${response.status})`);
  const index = await response.json() as SweIndex;
  if (index.schema_version !== 1 || !Array.isArray(index.tasks)) throw new Error('Unsupported task diagnosis schema');
  return { index, revision: sha };
}
async function sha256(bytes: BufferSource) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
}
export async function loadSweTrajectory(repo: string, revision: string, task: SweTask): Promise<SweTrajectory> {
  if (!task.trajectory || !task.trajectory_sha256) throw new Error('No completed trajectory');
  const response = await fetch(hfUrl(repo, revision, task.trajectory));
  if (!response.ok) throw new Error(`Cannot read trajectory (${response.status})`);
  const bytes = await response.arrayBuffer();
  if (await sha256(bytes) !== task.trajectory_sha256) throw new Error('Trajectory checksum mismatch. Highlighting withheld.');
  return JSON.parse(new TextDecoder().decode(bytes));
}
export function responseField(trajectory: SweTrajectory, span: LoopSpan): string {
  const msg = trajectory.messages[span.message_index];
  const raw = msg?.extra?.response?.choices?.[0]?.message ?? msg;
  const text = (raw as Record<string, unknown> | undefined)?.[span.field];
  if (typeof text !== 'string') throw new Error('The diagnosed response field is missing');
  return text;
}
export function splitLoop(text: string, span: LoopSpan) {
  const { start, unit_length: length, repeats } = span;
  if (![start, length, repeats].every(Number.isSafeInteger) || start < 0 || length <= 0 || repeats < 8 || start + length*repeats > text.length) throw new Error('Invalid loop boundaries');
  const unit = text.slice(start, start+length);
  const copies = Array.from({ length: repeats }, (_, i) => text.slice(start+i*length, start+(i+1)*length));
  if (!copies.every(copy => copy === unit)) throw new Error('Repeated blocks differ. Highlighting withheld.');
  return { before: text.slice(0, start), copies, after: text.slice(start+length*repeats) };
}
export async function verifyLoop(text: string, span: LoopSpan) {
  const split = splitLoop(text, span);
  const encode = new TextEncoder();
  if (await sha256(encode.encode(text)) !== span.text_sha256 || await sha256(encode.encode(split.copies[0])) !== span.unit_sha256) throw new Error('Loop text checksum mismatch');
  return split;
}
