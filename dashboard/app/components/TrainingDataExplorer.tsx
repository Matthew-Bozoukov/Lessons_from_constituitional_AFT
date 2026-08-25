"use client";

// ABOUTME: /datasets explorer: lists the org's training-data repos live from Hugging Face by
// ABOUTME: card tag, resolves each to a streamable corpus, and hands them to DatasetViewer.

import { useEffect, useMemo, useState } from "react";
import { CloudOff, LoaderCircle } from "lucide-react";
import { composition } from "@/lib/composition";
import { describeLoadError } from "@/lib/lazy";
import {
  TrainingDataRepo, corpusManifest, listTrainingData, repoDate, repoUrl,
} from "@/lib/trainingData";
import { DatasetViewer, DatasetViewerEntry } from "./DatasetViewer";

type Resolved = { repo: TrainingDataRepo; entry?: DatasetViewerEntry; reason?: string };

/** The repo name minus org and date, as words: the date sits in the summary. */
function titleFor(repo: TrainingDataRepo): string {
  const name = repo.repo.split("/")[1] || repo.repo;
  return name.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/[-_]+/g, " ");
}

/** Facts from the card tags, so a row is described without reading its README. */
function summaryFor(repo: TrainingDataRepo): string {
  const parts = [repo.kind];
  if (repo.pipeline) parts.push(repo.pipeline);
  parts.push(repo.constitution && repo.constitution !== "none"
    ? `constitution ${repo.constitution}`
    : "no constitution");
  if (repo.stage) parts.push(`${repo.stage} stage`);
  const date = repoDate(repo.repo) || repo.lastModified.slice(0, 10);
  if (date) parts.push(`published ${date}`);
  return parts.join(" · ");
}

export function TrainingDataExplorer({ org }: { org?: string }) {
  const [repos, setRepos] = useState<TrainingDataRepo[] | null>(null);
  const [listError, setListError] = useState("");
  const [resolved, setResolved] = useState<Resolved[] | null>(null);
  const [showSmoke, setShowSmoke] = useState(false);

  // -- discovery: one Hub listing, keyed on the training-data tag ------------
  useEffect(() => {
    let cancelled = false;
    setRepos(null);
    setResolved(null);
    setListError("");
    listTrainingData(org)
      .then((found) => { if (!cancelled) setRepos(found); })
      .catch((error) => { if (!cancelled) setListError(describeLoadError(error, "Hugging Face")); });
    return () => { cancelled = true; };
  }, [org]);

  // -- resolution: the rows file and stats for every repo, in parallel --------
  // A repo that resolves to nothing browsable is kept, with its reason, rather
  // than dropped: "N corpora publish no records" is a described gap, while a
  // silently shorter list reads as the org having fewer datasets.
  useEffect(() => {
    if (!repos) return;
    let cancelled = false;
    Promise.all(
      repos.map(async (repo): Promise<Resolved> => {
        try {
          const dataset = await corpusManifest(repo);
          return {
            repo,
            entry: {
              id: repo.repo,
              title: titleFor(repo),
              summary: summaryFor(repo),
              status: repo.smoke ? "smoke" : repo.kind,
              tags: repo.tags,
              dataset,
              mock: repo.mock,
            },
          };
        } catch (error) {
          return { repo, reason: describeLoadError(error, `Hugging Face (${repo.repo})`) };
        }
      }),
    ).then((rows) => { if (!cancelled) setResolved(rows); });
    return () => { cancelled = true; };
  }, [repos]);

  // Smoke runs are real pushes of a few rows made to exercise a pipeline, not
  // corpora anything trained on. They are folded away by default and counted,
  // never hidden outright.
  const visible = useMemo(
    () => (resolved || []).filter((r) => showSmoke || !r.repo.smoke),
    [resolved, showSmoke],
  );
  const smokeCount = (resolved || []).filter((r) => r.repo.smoke).length;
  const entries = visible.flatMap((r) => (r.entry ? [r.entry] : []));
  const unresolved = visible.filter((r) => !r.entry);
  const withBlend = entries.filter(
    (e) =>
      composition(e.dataset?.stats.categories, e.dataset?.stats.categories_source)
        ?.constitutionShare != null,
  ).length;
  const kinds = new Map<string, number>();
  for (const e of entries) kinds.set(e.status, (kinds.get(e.status) || 0) + 1);

  // -- render ---------------------------------------------------------------
  if (listError) return <div className="empty-state">{listError}</div>;
  if (!repos) {
    return (
      <div className="empty-state">
        <LoaderCircle size={13} className="spin" /> Listing training-data repos on Hugging Face…
      </div>
    );
  }
  if (!repos.length) {
    return (
      <div className="empty-state">
        No training-data repos found. A corpus appears here once its HF repo is public and
        its card carries the <code>training-data</code> tag — stamped automatically by
        <code> synth</code>, <code>mix</code> and <code>properties/ablate</code> pushes since
        2026-08-25; older repos need their cards backfilled.
      </div>
    );
  }
  if (!resolved) {
    return (
      <div className="empty-state">
        <LoaderCircle size={13} className="spin" /> Resolving {repos.length} corpora…
      </div>
    );
  }

  return (
    <div className="training-data-explorer">
      <div className="explorer-controls">
        <div className="controls-row">
          {/* What the list contains, in the terms the tags state it: nothing here
              is inferred from a repo's name. */}
          <p className="corpus-mix">
            <strong>{entries.length}</strong> corpora
            {[...kinds.entries()].map(([kind, n]) => (
              <span key={kind}> · {n} {kind}</span>
            ))}
            {" · "}<strong>{withBlend}</strong> publish their blend
            {smokeCount > 0 && <span> · {smokeCount} smoke runs {showSmoke ? "shown" : "hidden"}</span>}
          </p>
          {smokeCount > 0 && (
            <label className="compare-toggle">
              <input
                type="checkbox"
                checked={showSmoke}
                onChange={(event) => setShowSmoke(event.target.checked)}
              />
              show smoke runs
            </label>
          )}
        </div>
      </div>

      {/* Folded into one line: as full-width error blocks these read as things
          being broken, when what they record is a tagged repo that publishes no
          recognisable conversation file. The detail is one click away. */}
      {unresolved.length > 0 && (
        <details className="unresolved-note">
          <summary>
            <CloudOff size={14} />
            {unresolved.length} {unresolved.length === 1 ? "corpus publishes" : "corpora publish"} no
            records to browse
          </summary>
          <ul>
            {unresolved.map(({ repo, reason }) => (
              <li key={repo.repo}>
                <strong>{titleFor(repo)}</strong>
                {reason ? ` — ${reason}` : ""}{" "}
                <a href={repoUrl(repo.repo)} target="_blank" rel="noreferrer">
                  open on Hugging Face
                </a>
              </li>
            ))}
          </ul>
        </details>
      )}

      {entries.length > 0 ? (
        <DatasetViewer datasets={entries} />
      ) : (
        <div className="empty-state">
          Every discovered repo is a smoke run or publishes nothing browsable.
        </div>
      )}
    </div>
  );
}
