"use client";
// ABOUTME: Compare published dataset traits alongside downstream model evaluation outcomes.
// ABOUTME: Evidence labels and protocol checks keep observations separate from causal claims.
import { useEffect, useState } from "react";
import { artifactUrl, compatible, difference, listComparisons, loadComparison, traitDiffers,
  type Arm, type Comparison, type ComparisonRepo } from "@/lib/comparisons";

const number = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 2 });
const signed = (value: number) => `${value > 0 ? "+" : ""}${number(value)}`;
const failure = (error: unknown) => error instanceof Error ? error.message : "Could not load comparison";

function ArtifactLinks({ arm }: { arm: Arm }) {
  return <details className="comparison-provenance">
    <summary>Dataset → model → evaluation</summary>
    {[{kind:"Dataset", ...arm.dataset}, {kind:"Model", ...arm.model}, {kind:"Evaluation", ...arm.evaluation}].map(ref =>
      <p key={ref.kind}><strong>{ref.kind}</strong> <a href={artifactUrl(ref.repo, ref.revision, ref.kind === "Model")} target="_blank" rel="noreferrer">{ref.repo}</a><br /><code>{ref.revision}</code></p>)}
    <p>{arm.dataset.row_count.toLocaleString()} training rows · {arm.dataset.subset}</p>
    <p>Training seed: {arm.model.seed ?? "not recorded"} · {arm.evaluation.repeats} evaluation passes</p>
    <p>Base revision: <code>{arm.model.base_revision}</code></p>
  </details>;
}

export function ComparisonView({ data }: { data: Comparison }) {
  const [baselineId, setBaseline] = useState(data.arms[0].id);
  const [metricId, setMetric] = useState(data.metrics[0].id);
  const [differencesOnly, setDifferencesOnly] = useState(true);
  const baseline = data.arms.find(a=>a.id===baselineId)!;
  const metric = data.metrics.find(m=>m.id===metricId)!;
  const traits = data.traits.filter(t=>!differencesOnly || traitDiffers(data,t.id));
  const values = data.arms.flatMap(a=>a.evaluation.metrics[metricId] ? [a.evaluation.metrics[metricId].value] : []);
  const max=Math.max(1,...values.map(Math.abs));
  return <article className="comparison-detail">
    <header><h2>{data.title}</h2><p>{data.summary}</p></header>
    <div className="comparison-controls">
      <label>Reference model<select value={baselineId} onChange={e=>setBaseline(e.target.value)}>{data.arms.map(a=><option key={a.id} value={a.id}>{a.label}</option>)}</select></label>
      <label>Outcome<select value={metricId} onChange={e=>setMetric(e.target.value)}>{data.metrics.map(m=><option key={m.id} value={m.id}>{m.label}</option>)}</select></label>
    </div>
    <section aria-labelledby="comparison-outcomes"><h3 id="comparison-outcomes">Model outcomes</h3>
      <p className="table-note">{metric.label} ({metric.unit || "value"}){metric.lower_is_better === undefined ? "" : ` · ${metric.lower_is_better ? "Lower" : "Higher"} is better`}. Differences are candidate minus reference.</p>
      <div className="comparison-outcomes">
        {data.arms.map(arm=>{
          const score=arm.evaluation.metrics[metricId], delta=difference(data,baseline,arm,metricId);
          const unit=metric.unit === "%" ? "pp" : metric.unit;
          return <div className={`comparison-outcome ${arm.id===baselineId ? "is-reference" : ""}`} key={arm.id}>
            <div className="comparison-score"><h4>{arm.label}{arm.id===baselineId && <span>Reference</span>}</h4>
              <strong>{score ? `${number(score.value)}${metric.unit === "%" ? "%" : ` ${metric.unit}`}` : "Not reported"}</strong></div>
            {score && <><div className="comparison-bar" aria-hidden="true"><i style={{width:`${Math.abs(score.value)/max*100}%`}} /></div>
              <p>{score.denominator !== undefined && <span>{score.numerator}/{score.denominator} · </span>}{score.interval ? `95% CI [${number(score.interval.low)}, ${number(score.interval.high)}] · ${score.interval.method}` : "Uncertainty not reported"}</p></>}
            {arm.id!==baselineId && <p className="comparison-delta">{!compatible(baseline,arm) ? "Different evaluation protocols — difference unavailable" : delta ?
              <>{signed(delta.value)} {unit} versus reference{delta.interval ? ` · paired 95% CI [${signed(delta.interval.low)}, ${signed(delta.interval.high)}] (${delta.interval.method})` : " · paired uncertainty not reported"}</> : "Outcome missing — difference unavailable"}</p>}
            <ArtifactLinks arm={arm} />
          </div>;
        })}
      </div>
    </section>
    <section aria-labelledby="comparison-traits"><div className="comparison-section-head"><h3 id="comparison-traits">What changed in the datasets?</h3>
      <label><input type="checkbox" checked={differencesOnly} onChange={e=>setDifferencesOnly(e.target.checked)} /> Differences only</label></div>
      <p className="table-note">Measured = computed or audited on the stated population. Design = construction rule, not measured prevalence. Not measured does not mean absent.</p>
      <div className="comparison-table-wrap"><table className="comparison-table"><caption>Dataset properties and their evidence</caption><thead><tr><th scope="col">Property</th>{data.arms.map(a=><th scope="col" key={a.id}>{a.label}</th>)}</tr></thead>
        <tbody>{traits.map(t=><tr key={t.id}><th scope="row">{t.label}<small>{t.description}</small></th>{data.arms.map(a=>{
          const trait=a.traits[t.id];
          return <td key={a.id}><div>{trait?.value == null ? "Not measured" : typeof trait.value === "number" ? number(trait.value) : trait.value}</div>
            <span className={`comparison-basis ${trait?.basis || "unmeasured"}`}>{trait?.basis === "design" ? "Design" : trait?.basis === "measured" ? "Measured" : "Not measured"}</span>
            {trait?.evidence.map((e,i)=><a key={`${e.url}-${i}`} href={e.url} target="_blank" rel="noreferrer" className="comparison-evidence">{e.label} ↗</a>)}</td>;
        })}</tr>)}</tbody></table></div>
      {!traits.length && <p>No documented differences in these properties. Show all to inspect shared or unmeasured properties.</p>}
    </section>
    <section className="comparison-limitations"><h3>What this comparison can tell us</h3><ul>{data.limitations.map(l=><li key={l}>{l}</li>)}</ul></section>
    <details className="comparison-protocol"><summary>Exact evaluation settings</summary>{data.arms.map(a=><div key={a.id}><h4>{a.label}</h4><pre>{JSON.stringify(a.evaluation.protocol,null,2)}</pre></div>)}</details>
  </article>;
}

export function ComparisonExplorer() {
  const [repos,setRepos]=useState<ComparisonRepo[] | null>(null);
  const [selected,setSelected]=useState("");
  const [data,setData]=useState<Comparison | null>(null);
  const [error,setError]=useState("");
  const [attempt,setAttempt]=useState(0);
  useEffect(()=>{
    let cancelled=false;
    listComparisons().then(rows=>{if (!cancelled) {
      setRepos(rows);
      const requested=new URL(window.location.href).searchParams.get("repo");
      setSelected(rows.find(r=>r.repo===requested)?.repo || rows[0]?.repo || "");
    }}).catch(e=>{if (!cancelled) setError(failure(e));});
    return ()=>{cancelled=true;};
  },[attempt]);
  useEffect(()=>{
    const repo=repos?.find(r=>r.repo===selected);
    if (!repo) return;
    let cancelled=false;
    loadComparison(repo).then(value=>{if (!cancelled) setData(value);}).catch(e=>{if (!cancelled) setError(failure(e));});
    return ()=>{cancelled=true;};
  },[repos,selected,attempt]);
  return <>
    {repos && repos.length>0 && <label className="comparison-picker">Published comparison<select value={selected} onChange={e=>{
      setSelected(e.target.value);setData(null);setError("");
      const url=new URL(window.location.href);url.searchParams.set("repo",e.target.value);window.history.replaceState(null,"",url);
    }}>{repos.map(r=><option key={r.repo} value={r.repo}>{r.title}</option>)}</select></label>}
    {error ? <div role="alert" className="comparison-limitations"><p>{error}</p><button onClick={()=>{setError("");setData(null);setAttempt(a=>a+1);}}>Retry loading</button></div>
      : !repos ? <p role="status">Listing dataset-model comparisons on Hugging Face…</p>
      : !repos.length ? <p>No public dataset-model comparisons published yet.</p>
      : !data ? <p role="status">Loading the pinned comparison…</p> : <ComparisonView key={selected} data={data} />}
    {data && <p className="table-note"><a href={artifactUrl(selected,repos!.find(r=>r.repo===selected)!.revision)} target="_blank" rel="noreferrer">Published evidence and downloadable comparison ↗</a></p>}
  </>;
}
