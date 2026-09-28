// ABOUTME: Ensure grading, pending work and literal repetition are never conflated in the viewer.
// ABOUTME: Verify Unicode offsets, tampered evidence and original-text reconstruction.
import test from 'node:test';
import assert from 'node:assert/strict';
import { outcome, splitLoop, verifyLoop, loadSweTrajectory, loadSweIndex } from '../lib/swebench.ts';
import { createHash } from 'node:crypto';
const sha = s => createHash('sha256').update(s).digest('hex');
test('only officially graded valid outcomes are passes or failures', () => {
  const t = { status: 'valid', graded: true, resolved: true, loops: [{}] };
  assert.equal(outcome(t), 'passed');
  assert.equal(outcome({ ...t, resolved: false }), 'looped');
  assert.equal(outcome({ ...t, resolved: false, loops: [] }), 'other');
  assert.equal(outcome({ ...t, graded: false }), 'ungraded');
  assert.equal(outcome({ ...t, resolved: null }), 'ungraded');
  assert.equal(outcome({ ...t, status: 'pending' }), 'incomplete');
});
test('highlights preserve original text with Unicode and partial tails', async () => {
  const before = 'Before 🔬\n'; const unit = 'Repeated 🚀 passage\n\n'; const tail = 'Repeated';
  const text = before + unit.repeat(12) + tail;
  const span = { start: before.length, unit_length: unit.length, repeats: 12, unit_sha256: sha(unit), text_sha256: sha(text) };
  const s = await verifyLoop(text, span);
  assert.equal(s.before + s.copies.join('') + s.after, text);
  assert.equal(s.copies.length, 12);
  assert.throws(() => splitLoop(text.replace('passage', 'changed'), span), /differ/);
  assert.throws(() => splitLoop(text, { ...span, repeats: 300 }), /boundaries/);
  await assert.rejects(verifyLoop(text, { ...span, text_sha256: 'wrong' }), /checksum/);
});
test('trajectory checksum guards against stale or wrong attempt data', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('{"messages":[]}'));
  await assert.rejects(loadSweTrajectory('org/run', 'abc123', { trajectory: 'rollouts/task/attempt/checkpoint.traj.json', trajectory_sha256: 'bad' }), /checksum/);
});
test('task index uses resolved HF commit and missing diagnoses remain unknown', async t => {
  const urls = [];
  t.mock.method(globalThis, 'fetch', async url => {
    urls.push(url);
    return String(url).includes('/api/') ? Response.json({ sha: 'abc123' }) : Response.json({ schema_version: 1, tasks: [] });
  });
  assert.equal((await loadSweIndex('test-org/test-run')).revision, 'abc123');
  assert.match(urls[1], /\/resolve\/abc123\/metadata\/task-browser.json$/);
  t.mock.method(globalThis, 'fetch', async url => String(url).includes('/api/') ? Response.json({ sha: 'abc123' }) : new Response('', { status: 404 }));
  await assert.rejects(loadSweIndex('test-org/missing'), /Missing diagnoses are not classified/);
});
