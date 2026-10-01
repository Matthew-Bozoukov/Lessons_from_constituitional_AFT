"use client";
// ABOUTME: Browse graded passes, looped failures and other failures without downloading every transcript.
// ABOUTME: Show alternating, numbered copies sliced from the verified original response, never reconstructed text.
import { useEffect, useId, useMemo, useState } from 'react';
import { listEvalRuns, type EvalRun } from '@/lib/evalRuns';
import { hfUrl, loadSweIndex, loadSweTrajectory, outcome, responseField, verifyLoop, type LoopSpan, type Outcome, type SweIndex, type SweTask, type SweTrajectory } from '@/lib/swebench';

const labels: Record<Outcome, string> = { passed: 'Passed', looped: 'Failed · looped', other: 'Failed · other issues', ungraded: 'Awaiting grading', incomplete: 'Not completed' };
const failure = (e: unknown) => e instanceof Error ? e.message : 'Could not load evidence';

export function SweBenchDiscovery() {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [repo, setRepo] = useState('');
  const [error, setError] = useState('');
  useEffect(() => { let live = true;
    listEvalRuns().then(all => { if (live) setRuns(all.filter(r => r.tags?.includes('swebench-lite'))); }).catch(e => { if (live) setError(failure(e)); });
    return () => { live = false; };
  }, []);
  const selected = runs?.some(r => r.repo === repo) ? repo : runs?.[0]?.repo;
  return <section className="swe-browser">
    <label className="swe-run-picker">SWE-bench Lite run
      <select value={selected ?? ''} onChange={e => setRepo(e.target.value)} disabled={!runs?.length}>
        {!runs?.length && <option>{runs ? 'No public Lite runs found' : 'Finding runs on Hugging Face…'}</option>}
        {runs?.map(r => <option key={r.repo} value={r.repo}>{r.repo.split('/')[1]}</option>)}
      </select>
    </label>
    {error && <p role="alert">{error}</p>}
    {selected && <SweBenchBrowser key={selected} repo={selected} />}
  </section>;
}

export function SweBenchBrowser({ repo }: { repo: string }) {
  const [loaded, setLoaded] = useState<{ index: SweIndex; revision: string }>();
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<Outcome>('looped');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState('');
  const id = useId();
  useEffect(() => { let live = true;
    loadSweIndex(repo).then(d => { if (live) setLoaded(d); }).catch(e => { if (live) setError(failure(e)); });
    return () => { live = false; };
  }, [repo]);
  const tasks = useMemo(() => loaded?.index.tasks ?? [], [loaded]);
  const visible = tasks.filter(t => outcome(t) === filter && t.id.toLowerCase().includes(search.toLowerCase()));
  const task = visible.find(t => t.id === selected) ?? visible[0];
  if (error) return <p role="alert" className="empty-state">{error}</p>;
  if (!loaded) return <p role="status">Loading published task outcomes…</p>;
  return <section className="swe-browser" aria-label="SWE-bench task outcomes">
    <div className="swe-summary"><div><span className="eyebrow">Official tests + transcript evidence</span><h2>Where did the run end?</h2></div>
      <a href={`https://huggingface.co/datasets/${repo}/tree/${loaded.revision}`} target="_blank" rel="noreferrer">Pinned HF snapshot ↗</a></div>
    <div className="swe-outcomes" aria-label="Filter task outcomes">
      {(Object.keys(labels) as Outcome[]).map(key => <button key={key} type="button" aria-pressed={filter === key} className={`swe-outcome ${key}`} onClick={() => { setFilter(key); setSelected(''); }}>
        <strong>{tasks.filter(t => outcome(t) === key).length}</strong><span>{labels[key]}</span>
      </button>)}
    </div>
    <p className="swe-note">Passed means the official tests resolved the task, not merely that the agent submitted a patch. “Looped” means a failed task contains confirmed repeated text; it does not establish the cause of failure. Other issues means no loop meeting this detector’s threshold was found.</p>
    <div className="swe-task-controls"><label htmlFor={`${id}-search`}>Find an issue<input id={`${id}-search`} placeholder="Search task ID…" value={search} onChange={e => setSearch(e.target.value)} /></label>
      <label htmlFor={`${id}-task`}>{labels[filter]} · {visible.length} tasks<select id={`${id}-task`} value={task?.id ?? ''} onChange={e => setSelected(e.target.value)} disabled={!visible.length}>
        {!visible.length && <option>No matching tasks</option>}{visible.map(t => <option key={t.id} value={t.id}>{t.id}</option>)}
      </select></label></div>
    {task ? <TaskEvidence key={`${repo}:${task.id}`} repo={repo} revision={loaded.revision} task={task} /> : <p className="empty-state">No tasks in this category match your search.</p>}
    <details className="swe-method"><summary>How looping is identified</summary><p>{loaded.index.definition}</p><p>Only the final valid attempt determines a task’s outcome. Interrupted attempts and ungraded work are kept separate. A positive finding is evidence of literal repetition; a negative finding does not rule out shorter or paraphrased loops.</p></details>
  </section>;
}

function TaskEvidence({ repo, revision, task }: { repo: string; revision: string; task: SweTask }) {
  const [trajectory, setTrajectory] = useState<SweTrajectory>();
  const [error, setError] = useState('');
  const [loop, setLoop] = useState(0);
  const [history, setHistory] = useState(false);
  useEffect(() => { let live = true;
    if (task.trajectory) loadSweTrajectory(repo, revision, task).then(t => { if (live) setTrajectory(t); }).catch(e => { if (live) setError(failure(e)); });
    return () => { live = false; };
  }, [repo, revision, task]);
  return <article className="swe-evidence">
    <header><span className="eyebrow">{labels[outcome(task)]}</span><h3>{task.id}</h3><p>{task.reason?.replaceAll('_', ' ') || task.status} · {task.attempts} recorded attempt{task.attempts === 1 ? '' : 's'}</p>
      {task.trajectory && <a href={hfUrl(repo, revision, task.trajectory)} target="_blank" rel="noreferrer">Original trajectory JSON ↗</a>}</header>
    {task.discarded_response_corrections > 0 && <p className="swe-warning">{task.discarded_response_corrections} tool-format correction(s) have missing assistant responses. Those discarded responses cannot be assessed for looping.</p>}
    {error && <p role="alert">{error}</p>}
    {!task.trajectory && <p>This task has no completed model outcome. It is not counted as a model failure.</p>}
    {task.trajectory && !trajectory && !error && <p role="status">Fetching and verifying this trajectory…</p>}
    {trajectory && task.loops.length > 0 && <>
      {task.loops.length > 1 && <label>Repeated response<select value={loop} onChange={e => setLoop(Number(e.target.value))}>{task.loops.map((l, i) => <option key={i} value={i}>Message {l.message_index + 1} · {l.field}</option>)}</select></label>}
      <LoopEvidence key={loop} trajectory={trajectory} span={task.loops[loop]} />
    </>}
    {trajectory && <><button className="swe-action" onClick={() => setHistory(!history)} aria-expanded={history}>{history ? 'Hide' : 'Read'} complete conversation</button>
      {history && <div className="swe-history">{trajectory.messages.map((m, i) => <details key={i}><summary>Message {i+1} · {m.role}</summary>
        {(m.reasoning_content || m.reasoning) && <><h4>Reasoning</h4><pre>{m.reasoning_content || m.reasoning}</pre></>}
        {m.content && <pre>{m.content}</pre>}{m.tool_calls && <pre>{JSON.stringify(m.tool_calls, null, 2)}</pre>}
      </details>)}</div>}</>}
  </article>;
}

function LoopEvidence({ trajectory, span }: { trajectory: SweTrajectory; span: LoopSpan }) {
  const [split, setSplit] = useState<Awaited<ReturnType<typeof verifyLoop>>>();
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);
  useEffect(() => { let live = true;
    Promise.resolve().then(() => verifyLoop(responseField(trajectory, span), span)).then(s => { if (live) setSplit(s); }).catch(e => { if (live) setError(failure(e)); });
    return () => { live = false; };
  }, [trajectory, span]);
  if (error) return <p role="alert">{error}</p>;
  if (!split) return <p role="status">Verifying every repeated copy…</p>;
  const start = page*3;
  return <section className="swe-loop" aria-label="Verified repeated passage">
    <div className="swe-loop-head"><div><span className="eyebrow">One response. The same passage, again.</span><h4>{span.repeats} identical consecutive copies</h4><p>{(span.fraction*100).toFixed(1)}% of this {span.field === 'content' ? 'answer' : 'reasoning'} field’s characters · message {span.message_index+1}</p></div><span className="swe-verified">✓ All copies verified equal</span></div>
    <p>These are consecutive slices of the original response. Colors mark copy boundaries. There are no tool calls between these copies.</p>
    <details><summary>Text before the loop ({Array.from(split.before).length.toLocaleString()} characters)</summary><pre>{split.before || '(None)'}</pre></details>
    <div className="swe-loop-map" aria-label="Every repeated copy">{split.copies.map((_, i) => <button key={i} className={`copy-color-${i%3}`} title={`Jump to copy ${i+1}`} aria-label={`Jump to copy ${i+1}`} aria-current={i >= start && i < start+3 ? 'true' : undefined} onClick={() => setPage(Math.floor(i/3))}>{i+1}</button>)}</div>
    <nav className="swe-copy-nav" aria-label="Repeated copy pages"><button disabled={!page} onClick={() => setPage(page-1)}>← Previous copies</button><span>Copies {start+1}–{Math.min(start+3, span.repeats)} of {span.repeats}</span><button disabled={start+3 >= span.repeats} onClick={() => setPage(page+1)}>Next copies →</button></nav>
    <div className="swe-copies">{split.copies.slice(start, start+3).map((copy, i) => <section key={start+i} className={`swe-copy copy-color-${(start+i)%3}`}><h5>Copy {start+i+1} <span>{start+i === 0 ? 'Reference copy' : 'Identical to copy 1'}</span></h5><pre>{copy}</pre></section>)}</div>
    <details><summary>Text after the complete copies ({Array.from(split.after).length.toLocaleString()} characters)</summary><pre>{split.after || '(None)'}</pre></details>
    <details><summary>Verification details</summary><p>Original trajectory SHA-256, complete response-field SHA-256, and repeated-unit SHA-256 checked. Every displayed copy is sliced from its own position in the saved response and compared literally.</p><code>{span.unit_sha256}</code></details>
  </section>;
}
