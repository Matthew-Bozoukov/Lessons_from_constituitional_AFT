# ABOUTME: Staged, resumable fresh GPT-OSS control preparation, Tinker SFT and publication.
# ABOUTME: Uses the shared Harmony bridge; every paid stage has a token-cost ceiling and durable receipts.
from __future__ import annotations

import argparse
import collections
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from convert import convert_row, digest
from dotenv import load_dotenv
from omegaconf import OmegaConf


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows readers/virus scanners can momentarily hold the destination. Retry
    # only the atomic rename, never the already-paid provider request.
    for attempt in range(12):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt==11: raise
            time.sleep(min(.05*(attempt+1),.5))


def readrows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]


def saverows(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def provenance(cfg):
    return {"git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],cwd=ROOT,text=True).strip()),
            "command": subprocess.list2cmdline(sys.argv), "config": OmegaConf.to_container(cfg, resolve=True)}


def prepare(cfg, out):
    from src.infra.huggingface import hf_download
    source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    for name in ["mixture.jsonl", "mixture_stats.json", "README.md", "mixture_config.yaml", "filter_report.json", "run_meta.json"]:
        cached = hf_download(cfg.source_repo, name, repo_type="dataset", revision=cfg.source_revision)
        (source/name).write_bytes(Path(cached).read_bytes())
    rows, targets, audit = [], [], []
    for i, row in enumerate(readrows(source/"mixture.jsonl")):
        converted, traces, record = convert_row(row, i)
        rows.append(converted)
        targets.extend(traces)
        audit.append(record)
    saverows(out/"converted_unenriched.jsonl", rows)
    write(out/"backfill_targets.json", targets)
    write(out/"conversion_audit.json", {**provenance(cfg), "rows": len(rows),
        "kinds": dict(collections.Counter(a["kind"] for a in audit)), "records": audit,
        "source": {"repo": cfg.source_repo, "revision": cfg.source_revision}})
    audit_data(cfg, out, out/"converted_unenriched.jsonl", "prebackfill_audit")


def audit_data(cfg, out, path, name):
    from datasets import load_dataset
    from src.infra.endpoints.harmony import make_renderer, supervised_examples, TOKENIZER_REVISION
    rows = readrows(path)
    loaded = load_dataset("json", data_files=str(path), split="train")
    if len(loaded) != len(rows):
        raise ValueError("HF loader lost rows")
    for a, b in zip(rows, loaded):
        if a != b:
            raise ValueError("HF loader changed the canonical payload")
    renderer = make_renderer(cfg.reasoning)
    census = collections.defaultdict(lambda: collections.Counter())
    total = collections.Counter()
    examples = []
    lengths = []
    for i, row in enumerate(rows):
        rendered = supervised_examples(renderer, row, cfg.train.max_length)
        lengths.extend(len(e['input_ids'])+1 for e in rendered)
        count = {"rows": 1, "datums": len(rendered),
                 "processed_tokens": sum(len(e["input_ids"]) for e in rendered),
                 "supervised_tokens": int(sum(sum(e["weights"]) for e in rendered)),
                 "over_8192_datums": sum(len(e["input_ids"])+1 > 8192 for e in rendered)}
        total.update(count)
        census[row["source"]].update(count)
        if len(examples) < 12 and (row["source"] not in {x["source"] for x in examples} or row["source"] == "apigen_function_calling"):
            e = rendered[-1]
            examples.append({"row": i, "source": row["source"], "decoded": renderer.tokenizer.decode(e["input_ids"]),
                             "supervised": renderer.tokenizer.decode([t for t, w in zip(e["target_tokens"], e["weights"]) if w])})
        if i % 1000 == 0:
            print(f"audit {i}/{len(rows)}", flush=True)
    report = {**provenance(cfg), "dataset_sha256": digest(rows), "tokenizer_revision": TOKENIZER_REVISION,
              "total": dict(total), "by_source": {k: dict(v) for k,v in census.items()},
              "max_sequence_tokens":max(lengths),
              "estimated_training_usd": total["processed_tokens"]*0.737/1e6}
    write(out/f"{name}.json", report)
    write(out/f"{name}_examples.json", examples)
    print(json.dumps({k:v for k,v in report.items() if k in {"total", "estimated_training_usd"}}, indent=2))
    return report


def backfill(cfg, out):
    import tinker
    from src.infra.endpoints.harmony import make_renderer, render_prompt
    from src.infra.endpoints.openrouter import OpenRouterClient, provider_price
    from src.data.mixture.reasoning_backfill import judge_trace, JUDGE_PROMPT
    service = tinker.ServiceClient()
    sampler = service.create_sampling_client(base_model=cfg.base_model)
    renderer = make_renderer(cfg.reasoning)
    rows = readrows(out/"converted_unenriched.jsonl")
    targets = json.loads((out/"backfill_targets.json").read_text())
    if cfg.backfill.get("limit"):
        targets = targets[:int(cfg.backfill.limit)]
    log = out/"backfill_attempts.jsonl"
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (out/"backfill_receipts").glob("*.json")]
    if any(r["status"] in {"reserved", "sampled", "judge_reserved"} for r in records):
        raise RuntimeError("Uncertain paid backfill receipt needs reconciliation before resume")
    accepted = {(r["row"], r["turn"]):r for r in records if r.get("accepted")}
    charged = sum(r.get("target_cost_upper_usd", 0) for r in records)
    judge_charged = sum(r.get("judge_cost_upper_usd", r.get("judge", {}).get("judge_cost") or 0) for r in records)
    judge_price = provider_price(cfg.backfill.judge)
    if judge_price is None:
        raise ValueError("Judge pricing must be pinned before spending")
    lock = threading.Lock()
    # Cost reservations persist before dispatch; an uncertain request is not retried
    # for free merely because the client failed to receive its response.
    def job(target):
        nonlocal charged, judge_charged
        key = (target["row"], target["turn"])
        if key in accepted:
            return accepted[key]
        judge = OpenRouterClient()
        # The outer, receipted loop owns attempts; disable hidden retry spending.
        judge.chat = lambda **kwargs: OpenRouterClient.chat.__wrapped__(judge, **kwargs)
        row, turn = rows[key[0]], key[1]
        prompt_messages = row["messages"][:turn]
        prompt = render_prompt(renderer, prompt_messages, row.get("tools"))
        reservation = (len(prompt.to_ints())*.33 + cfg.backfill.max_tokens*.84)/1e6
        previous = [r for r in records if (r["row"],r["turn"])==key]
        rec = previous[-1] if previous else None
        for attempt in range(len(previous), cfg.backfill.attempts):
            rec = {**target, "attempt": attempt, "accepted": False, "target_cost_upper_usd": reservation,
                   "status": "reserved"}
            receipt = out/"backfill_receipts"/f"{key[0]}_{key[1]}_{attempt}.json"
            with lock:
                if charged + reservation > cfg.backfill.max_cost_usd:
                    raise RuntimeError("Backfill target budget exhausted")
                charged += reservation
                write(receipt, rec)
            try:
                sampled = sampler.sample(prompt, num_samples=1, sampling_params=tinker.SamplingParams(
                    max_tokens=cfg.backfill.max_tokens, temperature=cfg.backfill.temperature,
                    seed=cfg.seed + key[0]*10 + attempt, stop=renderer.get_stop_sequences())).result()
                tokens = sampled.sequences[0].tokens
                parsed, termination = renderer.parse_response(tokens)
                response = renderer.to_openai_message(parsed)
                trace = response.get("reasoning_content", "").strip()
                rec.update(status="sampled", response=response, output_tokens=len(tokens),
                           raw_tokens=tokens, prompt_tokens=prompt.to_ints(), trace=trace, termination=str(termination))
                actual = (len(prompt.to_ints())*.33+len(tokens)*.84)/1e6
                with lock:
                    charged -= reservation-actual
                    rec["target_cost_upper_usd"] = actual
                    write(receipt,rec)
                if trace and termination.is_stop_sequence and not response.get("tool_calls"):
                    question = "\n\n".join(f"[{m['role']}] {m.get('content') or ''}" for m in prompt_messages)
                    # UTF-8 bytes conservatively bound token count; include room for
                    # transport framing. Each receipt permits exactly one judge call.
                    judge_prompt = JUDGE_PROMPT.format(question=question, reasoning=trace, answer=row["messages"][turn]["content"])
                    judge_upper = ((len(judge_prompt.encode('utf-8'))+1024)*judge_price['in']+64*judge_price['out'])/1e6
                    with lock:
                        if judge_charged+judge_upper > cfg.backfill.max_judge_cost_usd:
                            raise RuntimeError("Backfill judging budget exhausted")
                        judge_charged += judge_upper
                        rec.update(status="judge_reserved", judge_cost_upper_usd=judge_upper)
                        write(receipt, rec)
                    verdict = judge_trace(judge, cfg.backfill.judge, question, trace, row["messages"][turn]["content"],max_chars=None)
                    with lock:
                        settled = verdict.get('judge_cost')
                        if settled is not None:
                            judge_charged += settled-judge_upper
                            rec['judge_cost_upper_usd'] = settled
                    rec["judge"] = verdict
                    rec["accepted"] = verdict.get("verdict") == "yes"
                with lock:
                    rec["status"] = "completed"
                    write(receipt, rec)
                    with log.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
                if rec["accepted"]:
                    return rec
            except Exception as exc:
                rec.update(status="request_error", error_type=type(exc).__name__)
                with lock:
                    write(receipt,rec)
                    with log.open("a",encoding="utf-8") as f:
                        f.write(json.dumps(rec,ensure_ascii=False)+"\n")
                print(f"backfill row={key[0]} attempt={attempt+1} error={type(exc).__name__}; reservation retained",flush=True)
        return rec
    with ThreadPoolExecutor(max_workers=int(cfg.backfill.concurrency)) as pool:
        for future in as_completed([pool.submit(job, t) for t in targets]):
            rec = future.result()
            if rec["accepted"]:
                accepted[(rec["row"],rec["turn"])] = rec
            print(f"backfill accepted={len(accepted)}/{len(targets)} target-cost-reserved=${charged:.3f}", flush=True)
    unresolved = [t for t in targets if (t["row"], t["turn"]) not in accepted]
    settled = [json.loads(p.read_text(encoding='utf-8')) for p in (out/'backfill_receipts').glob('*.json')]
    infrastructure = [t for t in unresolved if not any(r['row']==t['row'] and r['turn']==t['turn']
        and r['status']=='completed' for r in settled)]
    write(out/"backfill_report.json", {"accepted":len(accepted), "required":len(targets),
        "unresolved":unresolved, "disposition":str(cfg.backfill.unresolved_policy),
        "target_cost_upper_usd":charged,"judge_cost_upper_usd":judge_charged,
        "unresolved_infrastructure":infrastructure})
    if infrastructure:
        raise RuntimeError(f'{len(infrastructure)} turns have no completed attempts; reconcile before publication')
    if unresolved and cfg.backfill.unresolved_policy != "answer_only":
        raise RuntimeError(f"{len(unresolved)} reasoning turns need review; final dataset not published")
    if unresolved:
        print(f"Explicit disposition: {len(unresolved)} turns retain their original answer without a reasoning field",flush=True)
    if cfg.backfill.get("limit"):
        print("Backfill smoke complete; full dataset not yet enriched")
        return
    for (i,j), rec in accepted.items():
        rows[i]["messages"][j]["reasoning_content"] = rec["trace"]
    final = out/"dataset"
    final.mkdir(exist_ok=True)
    saverows(final/"mixture.jsonl", rows)
    stats = audit_data(cfg, out, final/"mixture.jsonl", "final_audit")
    stats["reasoning_traces"] = {"model":cfg.base_model, "family":"gptoss120b", "turns":len(accepted),
                                "judge":cfg.backfill.judge, "reasoning_effort":cfg.reasoning}
    write(final/"mixture_stats.json", stats)


def publish_data(cfg, out):
    from src.infra.huggingface import hf_api, hf_repo_id, card_markdown, training_data_tags, hf_download
    final = out/"dataset"
    report = json.loads((out/"backfill_report.json").read_text())
    targets=json.loads((out/'backfill_targets.json').read_text())
    if report['required'] != len(targets) or report['accepted']+len(report['unresolved']) != len(targets):
        raise RuntimeError('Backfill report is not the full planned corpus')
    if report.get('unresolved_infrastructure'):
        raise RuntimeError('Unresolved infrastructure failure is not a finalized dataset')
    if report["unresolved"] and report.get("disposition") != "answer_only":
        raise RuntimeError("Unresolved backfill is not a final dataset")
    before,after=readrows(out/'converted_unenriched.jsonl'),readrows(final/'mixture.jsonl')
    if len(before)!=len(after): raise ValueError('Row count changed')
    allowed={(t['row'],t['turn']) for t in targets}
    for i,(a,b) in enumerate(zip(before,after)):
        b=json.loads(json.dumps(b))
        for j,m in enumerate(b['messages']):
            if m.get('reasoning_content') and (i,j) not in allowed:
                raise ValueError('Reasoning was inserted outside selected turns')
            m['reasoning_content']=None
        if a!=b: raise ValueError(f'Prompt/answer/source/order changed in row {i}')
    # Recount with the exact current training renderer, even if the background
    # backfill process started before renderer qualification was finalized.
    stats=audit_data(cfg,out,final/'mixture.jsonl','final_audit')
    prior=json.loads((final/'mixture_stats.json').read_text())
    stats['reasoning_traces']=prior['reasoning_traces']
    write(final/'mixture_stats.json',stats)
    fields = {"experiment":"GPT-OSS nosynth control: checked native reasoning and Harmony tool conversion",
        "date_generated": date.today().isoformat(), "constitution":"claude_distilled_09_principles (inherited source filtering; no constitutional corpus added)",
        "source_repo": "teaching_claude_why_replication @ "+provenance(cfg)["git_sha"], "models":{"generator":cfg.base_model,"judge":cfg.backfill.judge},
        "generation_config":OmegaConf.to_container(cfg), "schema":"mixture.jsonl: messages, source, tools (JSON string)",
        "provenance":{**provenance(cfg),"parent":cfg.source_repo,"parent_revision":cfg.source_revision},
        "naming_exception":"User requested exact parent name plus -gpt-oss-120b; inherited date is not conversion date"}
    card = card_markdown(fields, front_matter={"configs":[{"config_name":"default","data_files":"mixture.jsonl","default":True}],
        "tags":training_data_tags("mixture","nosynth",fields["constitution"],extra=["gpt-oss","harmony"])})
    (final/"README.md").write_text(card, encoding="utf-8")
    for name in ["conversion_audit.json", "backfill_report.json", "final_audit.json", "final_audit_examples.json"]:
        (final/name).write_bytes((out/name).read_bytes())
    if (out/'backfill_restart.json').exists():
        shutil.copy2(out/'backfill_restart.json',final/'backfill_restart.json')
    shutil.copytree(out/'source',final/'parent_provenance',dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('mixture.jsonl'))
    shutil.copytree(out/'backfill_receipts',final/'backfill_receipts',dirs_exist_ok=True)
    OmegaConf.save(cfg,final/'conversion_config.yaml')
    write(final/'run_meta.json',provenance(cfg))
    for src in [Path(__file__),Path(__file__).with_name('convert.py'),ROOT/'src/infra/endpoints/harmony.py']:
        dst=final/'code'/src.name
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(src,dst)
    api, repo = hf_api(), hf_repo_id(cfg.destination_name)
    # Explicit user naming exception only: keep all card/schema/pin requirements.
    if api.repo_exists(repo, repo_type="dataset"):
        raise RuntimeError(f"Refusing to overwrite existing dataset {repo}")
    api.create_repo(repo, repo_type="dataset", private=False)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=final)
    sha = api.dataset_info(repo).sha
    downloaded = hf_download(repo,'mixture.jsonl',repo_type='dataset',revision=sha)
    if Path(downloaded).read_bytes() != (final/'mixture.jsonl').read_bytes():
        raise RuntimeError('Published dataset did not round-trip exactly')
    write(out/"published_dataset.json", {"repo":repo,"revision":sha})
    print(f"Published {repo}@{sha}")


def train(cfg, out):
    import tinker
    from src.infra.endpoints.harmony import make_renderer, supervised_examples, token_mean_datums, TOKENIZER_REVISION
    from src.infra.huggingface import hf_download
    from src.naming import model_name
    if (out/'trained_adapter.json').exists():
        print('Training already completed; use the recorded final checkpoint')
        return
    pin = json.loads((out/"published_dataset.json").read_text())
    rows = readrows(hf_download(pin["repo"], "mixture.jsonl", repo_type="dataset", revision=pin["revision"]))
    renderer = make_renderer(cfg.reasoning)
    ordered = list(range(len(rows)))
    random.Random(cfg.seed).shuffle(ordered)
    nsteps = math.ceil(len(rows)/cfg.train.batch_rows)
    if cfg.train.epochs != 1:
        raise ValueError("This pilot is one epoch only")
    estimate = json.loads((out/"final_audit.json").read_text())["estimated_training_usd"]
    if estimate > cfg.train.max_cost_usd:
        raise RuntimeError(f"Training estimate ${estimate} exceeds budget")
    service = tinker.ServiceClient()
    receipt = out/"train_state.json"
    budget_path = out/'train_requests.jsonl'
    spent = sum(r['reserved_usd'] for r in readrows(budget_path)) if budget_path.exists() else 0
    if receipt.exists():
        state = json.loads(receipt.read_text())
        if state["dataset"] != pin or state["config_hash"] != digest(OmegaConf.to_container(cfg)):
            raise RuntimeError("Resume config/data differs")
        client = service.create_training_client_from_state_with_optimizer(state["checkpoint"])
        start = state["step"]
        organism = state['organism']
    else:
        client = service.create_lora_training_client(base_model=cfg.base_model, rank=cfg.train.rank,
            seed=cfg.seed, train_mlp=True, train_attn=True, train_unembed=True)
        write(out/"tinker_model_info.json", client.get_info().model_dump(mode="json"))
        start = 0
        organism = model_name('gptoss120b',cfg.seed,'nosynth')
        checkpoint = client.save_state('step-0000').result().path
        write(receipt,{'step':0,'checkpoint':checkpoint,'dataset':pin,
                       'config_hash':digest(OmegaConf.to_container(cfg)),'organism':organism})
    for step in range(start,nsteps):
        examples = []
        for i in ordered[step*cfg.train.batch_rows:(step+1)*cfg.train.batch_rows]:
            examples.extend(supervised_examples(renderer, rows[i], cfg.train.max_length))
        datums = token_mean_datums(examples)
        warmup = max(1, math.ceil(nsteps*cfg.train.warmup_ratio))
        scale = (step+1)/warmup if step < warmup else .5*(1+math.cos(math.pi*(step-warmup)/(nsteps-warmup)))
        lr = cfg.train.lr*scale
        processed = sum(len(e['input_ids']) for e in examples)
        reservation = processed*.737/1e6
        if spent+reservation > cfg.train.max_cost_usd:
            raise RuntimeError('Training budget exhausted including replayed/uncertain steps')
        with budget_path.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'step':step+1,'reserved_usd':reservation,'checkpoint_before_resume':start})+'\n')
        spent += reservation
        result = client.forward_backward(datums, loss_fn="cross_entropy").result()
        loss = result.metrics.get("loss:sum")
        if loss is None or not math.isfinite(float(loss)):
            raise RuntimeError(f"Missing/nonfinite training loss at step {step}")
        adam=tinker.AdamParams(learning_rate=lr, weight_decay=cfg.train.weight_decay,
            beta1=cfg.train.beta1,beta2=cfg.train.beta2,eps=cfg.train.adam_eps,
            grad_clip_norm=cfg.train.grad_clip_norm)
        opt = client.optim_step(adam).result()
        if any(not math.isfinite(float(v)) for v in (opt.metrics or {}).values()):
            raise RuntimeError(f'Nonfinite optimizer metric at step {step+1}')
        record = {"step":step+1,"loss":loss,"lr":lr,"metrics":result.metrics,
                  "optimizer_metrics":opt.metrics, "optimizer_params":adam.model_dump(),
                  "processed_tokens":sum(len(e["input_ids"]) for e in examples),
                  "supervised_tokens":sum(sum(e["weights"]) for e in examples)}
        with (out/"training_curve.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(record)+"\n")
        print(f"train {step+1}/{nsteps} loss={loss:.5f}",flush=True)
        if step == 0 or (step+1)%cfg.train.save_steps == 0 or step+1 == nsteps:
            checkpoint = client.save_state(f"step-{step+1:04d}").result().path
            write(receipt, {"step":step+1,"checkpoint":checkpoint,"dataset":pin,
                            "config_hash":digest(OmegaConf.to_container(cfg)),"organism":organism})
    sampler = client.save_weights_for_sampler(organism).result().path
    write(out/"trained_adapter.json", {**provenance(cfg),"sampler":sampler,"dataset":pin,
         "base_model":cfg.base_model,"tokenizer_revision":TOKENIZER_REVISION,"thinking":True,
         "backend":"tinker","precision":"provider managed; not independently verified",
         "organism":organism,"training_cost_upper_usd":spent,"steps":nsteps})
    print(f"Sampler checkpoint: {sampler}")


def export(cfg,out):
    from safetensors import safe_open
    import torch
    from tinker_cookbook.weights import download
    from src.infra.huggingface import hf_api,hf_repo_id,push_run_dir,hf_download
    meta=json.loads((out/'trained_adapter.json').read_text())
    launch=json.loads((out/'training_launch_observation.json').read_text())
    training_revision=subprocess.check_output(
        ['git','rev-parse',launch['training_code_commit']],text=True).strip()
    meta={**meta,'training_code_revision':training_revision,
          'export_code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    final=out/'adapter'
    if not (final/'adapter_model.safetensors').exists():
        download(tinker_path=meta['sampler'],output_dir=str(final))
    config=json.loads((final/'adapter_config.json').read_text())
    if config['r'] != cfg.train.rank:
        raise ValueError('Exported rank differs from requested training rank')
    tensors={}
    with safe_open(final/'adapter_model.safetensors',framework='pt') as f:
        for key in f.keys():
            tensor=f.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite exported tensor: {key}')
            tensors[key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),'numel':tensor.numel()}
    scopes={name:any(name in k for k in tensors) for name in ['attn','experts','unembed']}
    write(final/'tensor_audit.json',{'tensors':tensors,'parameter_count':sum(t['numel'] for t in tensors.values()),
        'name_scope_observations':scopes,'format':'Tinker native LoRA; external serving unqualified'})
    write(final/'training_meta.json',{**meta,'adapter_config':config,'loss_reduction':'token_mean',
        'model':cfg.base_model,'data_repo':meta['dataset']['repo'],'data_revision':meta['dataset']['revision']})
    OmegaConf.save(cfg,final/'train_config.yaml')
    for name in ['training_curve.jsonl','train_requests.jsonl','train_state.json','tinker_model_info.json',
                 'final_audit.json','final_audit_examples.json','backfill_report.json']:
        shutil.copy2(out/name,final/name)
    if (out/'training_launch_observation.json').exists():
        shutil.copy2(out/'training_launch_observation.json',final/'training_launch_observation.json')
    fields={'experiment':'Fresh GPT-OSS-120B nosynth control, one epoch of token-weighted Tinker SFT',
        'date_generated':meta['organism'][:10],
        'constitution':'claude_distilled_09_principles (inherited filtering; no constitutional corpus added)',
        'source_repo':'teaching_claude_why_replication @ '+training_revision,
        'models':{'base':cfg.base_model,'sampler':meta['sampler']},'generation_config':meta['config'],
        'schema':'Native Tinker LoRA matrices and config; use immutable Tinker sampler for inference. External PEFT/vLLM equivalence has not been tested.',
        'provenance':meta}
    api,repo=hf_api(),hf_repo_id(meta['organism'])
    if api.repo_exists(repo,repo_type='model'):
        raise RuntimeError(f'Refusing to overwrite existing adapter {repo}')
    push_run_dir(final,repo,fields,repo_type='model',front_matter={'base_model':cfg.base_model,
        'tags':['lora','tinker','gpt-oss','nosynth'],'datasets':[meta['dataset']['repo']]})
    info=api.model_info(repo,files_metadata=True)
    sha=info.sha
    def file_sha(path):
        with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
    remote=next(s for s in info.siblings if s.rfilename=='adapter_model.safetensors')
    if remote.lfs is None or remote.lfs.sha256!=file_sha(final/'adapter_model.safetensors'):
        raise RuntimeError('Published adapter hash mismatch')
    write(out/'published_adapter.json',{'repo':repo,'revision':sha,'sampler':meta['sampler']})
    print(f'Published adapter {repo}@{sha}')


def evaluate(cfg,out,smoke=False):
    meta=json.loads((out/'trained_adapter.json').read_text())
    qualification=out/'adapter_transport'/'passed.json'
    if not qualification.exists() or json.loads(qualification.read_text()).get('checkpoint')!=meta['sampler']:
        raise RuntimeError('Adapter transport smoke must pass first')
    root=out/('odcv_smoke' if smoke else 'odcv')
    spent=sum(r.get('reserved_usd',0) for p in root.glob('**/sampling.jsonl') for r in readrows(p))
    remaining=(1 if smoke else cfg.eval.max_target_cost_usd)-spent
    if remaining<=0: raise RuntimeError('ODCV target budget exhausted including prior runs')
    adapter_path=out/'published_adapter.json'
    if adapter_path.exists():
        adapter=json.loads(adapter_path.read_text())
    elif smoke:
        # This local --no-push smoke can overlap the large HF upload. Its target
        # is already pinned by the immutable sampler path; never invent an HF pin.
        adapter={'sampler':meta['sampler'],'publication_status':'pending at local smoke'}
    else:
        raise RuntimeError('Full evaluation requires a verified published adapter')
    protocol=OmegaConf.merge(OmegaConf.load('configs/eval/odcv/lite.yaml'),{
        'compare_published':False,'published_key':None,'concurrency':cfg.eval.concurrency,
        'prune_images':False,'prune_networks':False,'require_clean_pass':True,'smoke':smoke,
        'output_root':str(root),
        'tinker':{'reasoning':cfg.reasoning,'max_tokens':cfg.eval.max_tokens,'bind':'0.0.0.0',
                  'max_cost_usd':remaining,'adapter_artifact':adapter},
        'judge_budget':{'ledger':str(out/('smoke_judge_budget.json' if smoke else 'odcv_judge_budget.json')),
            'cap_usd':1 if smoke else cfg.eval.max_judge_cost_usd,'max_tokens':cfg.eval.judge_max_tokens}})
    path=out/('odcv_smoke_config.yaml' if smoke else 'odcv_config.yaml')
    OmegaConf.save(protocol,path)
    args=[sys.executable,'-m','src.eval.run_eval','--name','odcv','--config',str(path),
          '--port',str(cfg.eval.port),'--target',meta['sampler']]
    if smoke: args.append('--no-push')
    subprocess.run(args,check=True,cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare","backfill","publish-data","train","export","eval-smoke","eval"])
    parser.add_argument("--config", default="scratch/gptoss_control/config.yaml")
    parser.add_argument("--limit",type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    common = Path(subprocess.check_output(["git","rev-parse","--git-common-dir"], text=True).strip())
    load_dotenv(common.resolve().parent/".env")
    cfg = OmegaConf.load(args.config)
    if args.limit:
        cfg.backfill.limit = args.limit
    out = ROOT/cfg.output
    out.mkdir(parents=True, exist_ok=True)
    {"prepare":prepare,"backfill":backfill,"publish-data":publish_data,"train":train,"export":export,
     "eval-smoke":lambda c,o:evaluate(c,o,True),"eval":evaluate}[args.stage](cfg,out)


if __name__ == "__main__":
    main()
