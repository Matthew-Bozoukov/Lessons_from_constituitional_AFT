# ABOUTME: Offline coverage, duplicate census and deterministic condition-masked data audit packets.
# ABOUTME: Reuses corpus word/ngram helpers; emits evidence without filtering or making model calls.
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data.synth.check_corpus import ngrams, words


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rank(salt, value):
    return digest(f"{salt}:{value}".encode())


def make_audit(source, destination, *, sample=32, domain_field="domain",
               b_field="comparative", c_field="execution",
               salt="nonmoral-independent-audit-20260909", near_threshold=0.65):
    blob = Path(source).read_bytes()
    rows = [json.loads(line) for line in blob.decode("utf-8").splitlines() if line.strip()]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "freeze.json").exists():
        raise ValueError("Audit packet already frozen; use a new directory for a new snapshot")
    ids = [str(r["scenario_id"]) for r in rows]
    duplicate_ids = {k: v for k, v in Counter(ids).items() if v > 1}
    if duplicate_ids:
        raise ValueError(f"Nonunique IDs cannot form an unambiguous audit: {duplicate_ids}")
    buckets = defaultdict(list)
    exact, number_skeleton = defaultdict(list), defaultdict(list)
    token_sets = []
    incomplete = []
    for i, row in enumerate(rows):
        buckets[str(row.get(domain_field, "<missing>"))].append(i)
        missing = [k for k in ("user", "answer", b_field, c_field)
                   if not isinstance(row.get(k), str) or not row[k].strip()]
        if missing:
            incomplete.append({"scenario_id": ids[i], "missing_fields": missing})
        normalized = " ".join(words(str(row.get("user", ""))))
        exact[normalized].append(ids[i])
        number_skeleton[re.sub(r"\d+(?:\.\d+)?", "<number>", normalized)].append(ids[i])
        token_sets.append(ngrams(words(normalized), 4))
    near = []
    for i, left in enumerate(token_sets):
        for j in range(i + 1, len(rows)):
            right = token_sets[j]
            shared = len(left & right)
            similarity = shared / max(1, len(left) + len(right) - shared)
            if shared >= 12 and similarity >= near_threshold:
                near.append({"left": ids[i], "right": ids[j],
                             "word_4gram_jaccard": round(similarity, 4)})
    ordered = {k: sorted(v, key=lambda i: rank(salt, ids[i]))
               for k, v in sorted(buckets.items())}
    chosen = []
    # Round-robin stratification prevents large domains consuming the entire audit.
    while len(chosen) < min(sample, len(rows)):
        progressed = False
        for indices in ordered.values():
            if indices and len(chosen) < min(sample, len(rows)):
                chosen.append(indices.pop(0))
                progressed = True
        if not progressed:
            break
    chosen.sort(key=lambda i: rank(salt + ":display", ids[i]))
    packet, mapping = [], []
    for position, i in enumerate(chosen, 1):
        row = rows[i]
        forward = int(rank(salt + ":order", ids[i])[-1], 16) % 2 == 0
        x, y = (b_field, c_field) if forward else (c_field, b_field)
        audit_id = f"case_{position:03}"
        packet.append({"audit_id": audit_id, "system": row.get("original_system", ""),
                       "user": row.get("user", ""), "answer": row.get("answer", ""),
                       "text_x": row.get(x, ""), "text_y": row.get(y, "")})
        mapping.append({"audit_id": audit_id, "scenario_id": ids[i],
                        "domain": str(row.get(domain_field, "<missing>")),
                        "text_x_field": x, "text_y_field": y})
    census = {"rows": len(rows), "domain_counts": dict(Counter(
        str(r.get(domain_field, "<missing>")) for r in rows)),
        "incomplete": incomplete,
        "normalized_exact_prompt_groups": [v for k, v in exact.items() if k and len(v) > 1],
        "number_normalized_prompt_groups": [v for k, v in number_skeleton.items() if k and len(v) > 1],
        "near_prompt_pairs": near,
        "near_threshold": near_threshold,
        "interpretation": "Lexical duplicate signals require inspection; boilerplate or number-varied tasks are not automatically invalid. No rows filtered."}
    packet_blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in packet).encode()
    (destination / "review_packet.jsonl").write_bytes(packet_blob)
    (destination / "condition_key.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (destination / "coverage_duplicates.json").write_text(json.dumps(census, indent=2), encoding="utf-8")
    freeze = {"source": str(Path(source).resolve()), "source_sha256": digest(blob),
              "review_packet_sha256": digest(packet_blob), "original_denominator": len(rows),
              "audit_count": len(packet), "salt": salt, "domain_field": domain_field,
              "selection": "Metadata-stratified hash ordering frozen before manual review; no quality or ODCV selection.",
              "blinding": "Packet omits condition labels, IDs, generator judgements and revision history; reasoning content may reveal condition. Key kept separately.",
              "role": "Production quality audit may inspect training candidates; not a held-out benchmark. Fresh development32 remain excluded from SFT separately."}
    (destination / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    return freeze, census


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sample", type=int, default=32)
    parser.add_argument("--domain-field", default="domain")
    parser.add_argument("--b-field", default="comparative")
    parser.add_argument("--c-field", default="execution")
    args = parser.parse_args()
    if args.sample <= 0:
        parser.error("--sample must be positive")
    freeze, census = make_audit(args.source, args.destination, sample=args.sample,
                              domain_field=args.domain_field, b_field=args.b_field,
                              c_field=args.c_field)
    print(json.dumps({"freeze": freeze, "census": census}, indent=2))


if __name__ == "__main__":
    main()
