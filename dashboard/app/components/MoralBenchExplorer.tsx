"use client";

// ABOUTME: Side-by-side MoralBench comparison across any number of published runs, read
// ABOUTME: live from HF: four blocks overall, or one block broken out by moral foundation.

import { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, LoaderCircle } from "lucide-react";
import { describeLoadError } from "@/lib/lazy";
import { EvalRun, listEvalRuns, repoDate, repoUrl } from "@/lib/evalRuns";
import {
  BLOCKS, BLOCK_LABEL, CHANCE, FOUNDATIONS, FOUNDATION_COLOR, FOUNDATION_LABEL,
  type Block, type BlockKey, type Foundation, type MoralBenchRun,
  barFraction, belowChance, loadMoralBenchRun,
} from "@/lib/moralbench";

// Per-RUN series colours (the foundation palette is reserved for foundations, so the two
// legends never collide). Six is a deliberate cap on selection: past that, grouped bars
// stop being readable and the table is the better tool anyway.
const SERIES = [
  "var(--cyan)", "var(--violet)", "var(--lime)", "var(--amber)", "var(--red)", "var(--cyan-dim)",
];
const MAX_SELECTED = 6;

type Loaded = { run?: MoralBenchRun; error?: string };

function pct(x: number | null): string {
  return x === null ? "—" : `${(x * 100).toFixed(1)}%`;
}
function fmt(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(2);
}

/** One horizontal bar: where an arm sits between a block's floor and its ceiling. */
function Bar({ block, color, chance }: { block: Block; color: string; chance?: number }) {
  const frac = barFraction(block);
  // The chance baseline is drawn in the SAME coordinate space as the bar, so "below
  // chance" is something you see rather than something you compute.
  const chanceFrac =
    chance !== undefined && block.max > block.min
      ? Math.max(0, Math.min(1, (chance - block.min) / (block.max - block.min)))
      : null;
  return (
    <div className="mb-track">
      <div className="mb-fill" style={{ width: `${frac * 100}%`, background: color }} />
      {chanceFrac !== null && (
        <div className="mb-chance" style={{ left: `${chanceFrac * 100}%` }} title={`chance = ${chance}`} />
      )}
    </div>
  );
}

function BlockRow({
  label, entries, chance, note,
}: {
  label: string;
  entries: Array<{ key: string; name: string; color: string; block: Block }>;
  chance?: number;
  note?: string;
}) {
  if (!entries.length) return null;
  const sample = entries[0].block;
  return (
    <div className="mb-block">
      <div className="mb-block-head">
        <h4>{label}</h4>
        <span className="mb-range">
          reachable {fmt(sample.min)}–{fmt(sample.max)}
          {sample.maxDeterministic !== undefined && (
            <> · deterministic ceiling {fmt(sample.maxDeterministic)}</>
          )}
          {chance !== undefined && <> · chance {chance}</>}
          {" · "}n={sample.nItems}
        </span>
      </div>
      {note && <p className="mb-note">{note}</p>}
      {entries.map((e) => (
        <div className="mb-row" key={e.key}>
          <span className="mb-name" title={e.name}>{e.name}</span>
          <Bar block={e.block} color={e.color} chance={chance} />
          <span className="mb-val">
            {pct(e.block.normalized)}
            <em>{fmt(e.block.total)}</em>
            {chance !== undefined && belowChance(
              // key is only used to look the chance value back up; label carries the block
              (Object.keys(CHANCE) as BlockKey[]).find((k) => CHANCE[k] === chance) as BlockKey,
              e.block,
            ) && <b className="mb-flag">below chance</b>}
          </span>
        </div>
      ))}
    </div>
  );
}

export function MoralBenchExplorer() {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [listError, setListError] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [loaded, setLoaded] = useState<Record<string, Loaded>>({});
  const [view, setView] = useState<"overall" | "foundations">("overall");
  const [block, setBlock] = useState<BlockKey>("MFQ_binary");

  useEffect(() => {
    let cancelled = false;
    listEvalRuns()
      .then((all) => {
        if (cancelled) return;
        const mine = all.filter((r) => r.evalName === "moralbench");
        setRuns(mine);
        // Preselect the two most recent so the page is useful on arrival — the whole
        // point of this view is a comparison, and one run is not one.
        setSelected(mine.slice(0, 2).map((r) => r.repo));
      })
      .catch((e) => {
        if (!cancelled) setListError(describeLoadError(e, "Hugging Face"));
      });
    return () => { cancelled = true; };
  }, []);

  // No synchronous placeholder: a repo with no entry in `loaded` IS the loading state
  // (same convention as EvalRunExplorer). `cached()` dedupes by key, so the effect
  // re-firing before a promise resolves costs nothing.
  useEffect(() => {
    for (const repo of selected) {
      if (loaded[repo]) continue;
      loadMoralBenchRun(repo)
        .then((run) =>
          setLoaded((s) => ({
            ...s,
            [repo]: run ? { run } : { error: "not a MoralBench result" },
          })),
        )
        .catch((e) =>
          setLoaded((s) => ({
            ...s, [repo]: { error: describeLoadError(e, `Hugging Face (${repo})`) },
          })));
    }
  }, [selected, loaded]);

  const byRepo = useMemo(() => new Map((runs || []).map((r) => [r.repo, r])), [runs]);

  const series = useMemo(
    () =>
      selected
        .map((repo, i) => ({
          repo,
          color: SERIES[i % SERIES.length],
          name: [byRepo.get(repo)?.model || repo.split("/")[1], repoDate(repo)]
            .filter(Boolean).join(" · "),
          state: loaded[repo] || {},
        }))
        .filter((s) => s.state.run),
    [selected, loaded, byRepo],
  );

  function toggle(repo: string) {
    setSelected((s) =>
      s.includes(repo)
        ? s.filter((r) => r !== repo)
        : s.length >= MAX_SELECTED ? s : [...s, repo],
    );
  }

  if (listError) return <p className="empty-state">{listError}</p>;
  if (!runs) {
    return <p className="empty-state"><LoaderCircle className="spin" size={16} /> Finding MoralBench runs…</p>;
  }
  if (!runs.length) {
    return (
      <p className="empty-state">
        No MoralBench runs published yet. A run appears here once its HF repo is public
        and its card carries the <code>eval:moralbench</code> tag — stamped automatically
        by <code>uv run evals --name moralbench</code>.
      </p>
    );
  }

  const pending = selected.filter((r) => !loaded[r]);
  const failed = selected.filter((r) => loaded[r]?.error);

  return (
    <div className="mb-explorer">
      <div className="mb-pickers">
        <div className="control-group grow">
          <label>Runs to compare <span className="mb-hint">({selected.length}/{MAX_SELECTED})</span></label>
          <div className="mb-runlist">
            {runs.map((r) => {
              const on = selected.includes(r.repo);
              return (
                <button
                  key={r.repo}
                  type="button"
                  className="mb-runchip"
                  aria-pressed={on}
                  disabled={!on && selected.length >= MAX_SELECTED}
                  onClick={() => toggle(r.repo)}
                  style={on ? { borderColor: SERIES[selected.indexOf(r.repo) % SERIES.length] } : undefined}
                >
                  <span
                    className="mb-swatch"
                    style={{ background: on ? SERIES[selected.indexOf(r.repo) % SERIES.length] : "transparent" }}
                  />
                  {r.model}
                  <em>{r.mode}</em>
                  <span className="mb-date">{repoDate(r.repo)}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="control-group">
          <label>View</label>
          <div className="mb-tabs">
            <button type="button" aria-pressed={view === "overall"} onClick={() => setView("overall")}>
              Overall
            </button>
            <button type="button" aria-pressed={view === "foundations"} onClick={() => setView("foundations")}>
              By foundation
            </button>
          </div>
        </div>

        {view === "foundations" && (
          <div className="control-group">
            <label htmlFor="mb-block">Block</label>
            <select id="mb-block" value={block} onChange={(e) => setBlock(e.target.value as BlockKey)}>
              {BLOCKS.map((b) => <option key={b} value={b}>{BLOCK_LABEL[b]}</option>)}
            </select>
          </div>
        )}
      </div>

      <p className="mb-caveat">
        Bars show where an arm sits in the <strong>reachable</strong> range, not 0–100%:
        both binary options score, so the floor is 60% of the ceiling on MFQ and 74% on
        MFV, and a raw total hides most of the difference. <strong>Higher is not
        better</strong> — the score rewards agreement with the MFQ/MFV human norming
        sample, so an arm that downweights tradition scores lower by design. Binary and
        comparative share no scale and are never combined.
      </p>

      {pending.length > 0 && (
        <p className="empty-state"><LoaderCircle className="spin" size={16} /> Loading {pending.length} run(s)…</p>
      )}
      {failed.map((r) => (
        <p className="empty-state" key={r}>{r}: {loaded[r]?.error}</p>
      ))}

      {series.length === 0 ? (
        <p className="empty-state">Select at least one run.</p>
      ) : view === "overall" ? (
        <>
          {BLOCKS.map((key) => (
            <BlockRow
              key={key}
              label={BLOCK_LABEL[key]}
              chance={CHANCE[key]}
              note={
                key === "MFV_comparative"
                  ? "Two items share a prompt with opposite labels, so a model answering identically cannot exceed 23 of 24 (see the eval's NOTICE)."
                  : undefined
              }
              entries={series
                .map((s) => ({ key: s.repo, name: s.name, color: s.color, block: s.state.run!.blocks[key] }))
                .filter((e): e is { key: string; name: string; color: string; block: Block } => !!e.block)}
            />
          ))}
        </>
      ) : (
        <>
          {FOUNDATIONS.map((f) => {
            const entries = series
              .map((s) => ({
                key: s.repo, name: s.name, color: s.color,
                block: s.state.run!.blocks[block]?.byFoundation[f as Foundation],
              }))
              .filter((e): e is { key: string; name: string; color: string; block: Block } => !!e.block);
            if (!entries.length) return null;
            return (
              <div className="mb-foundation" key={f} style={{ borderLeftColor: FOUNDATION_COLOR[f] }}>
                <BlockRow label={FOUNDATION_LABEL[f]} entries={entries} />
              </div>
            );
          })}
          <p className="mb-note">
            Four items per foundation. A per-foundation number is coarse on its own —
            read the difference between arms on identical items, not one arm&apos;s level.
          </p>
        </>
      )}

      {series.length > 0 && (
        <div className="mb-health">
          <h4>Run health</h4>
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th><th>Mode</th><th>Reps</th><th>Swap</th>
                <th>Parse rate</th><th>Invalid</th><th>Answers A / B</th><th>Repo</th>
              </tr>
            </thead>
            <tbody>
              {series.map((s) => {
                const run = s.state.run!;
                return (
                  <tr key={s.repo}>
                    <td><span className="mb-swatch" style={{ background: s.color }} /> {s.name}</td>
                    <td>{run.mode || "—"}</td>
                    <td>{run.repetitions}</td>
                    <td>{run.swapOptions ? "swapped" : "released"}</td>
                    <td>{pct(run.parseRate)}</td>
                    <td className={run.invalidRate !== null && run.invalidRate > 0.05 ? "mb-warn" : ""}>
                      {pct(run.invalidRate)}
                    </td>
                    <td>{run.answerBalance ? `${run.answerBalance.A} / ${run.answerBalance.B}` : "—"}</td>
                    <td>
                      <a href={repoUrl(s.repo)} target="_blank" rel="noreferrer" className="legend-chip">
                        HF <ArrowUpRight size={12} />
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mb-note">
            An unparsed answer scores zero, which is below every reachable binary score —
            so a run with a high invalid rate can undershoot a block&apos;s own floor.
            Check this column before reading a low bar as a moral finding.
          </p>
        </div>
      )}
    </div>
  );
}
