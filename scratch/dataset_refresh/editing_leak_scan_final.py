# ABOUTME: Screens effective accepted conversations for possible author/revision-process references.
# ABOUTME: Keywords are flags only; actual human draft references require independent contextual adjudication.
import json
import re
from collections import Counter
from pathlib import Path
import run as runtime

base = Path(__file__).resolve().parents[2]
roots = [base/'output/2026-09-15_dataset_refresh_sonnet_qualified', base/'output/2026-09-15_dataset_refresh_diverse']
patterns = [r'\bdraft\b.{0,110}\b(?:answer|response|reasoning|surfaced|sound|correct|preserv\w*|repair\w*|revision|actual advice)\b', r'\b(?:original|previous|prior|revised|revision|earlier)\s+(?:answer|response|reasoning)\b', r'\b(?:system prompt|training prompt|hidden preference|provided preference|craft preference|trait text|the revision should)\b']
flags=[]
counts=Counter()
for root in roots:
    for arm in ['nonmoral-advice','da-lowstakes-refresh']:
        for path in sorted((root/arm/'records').glob('*/result.json')):
            data=runtime.load_result(path)
            if data['status']!='accepted': continue
            counts[root.name+'/'+arm]+=1
            record=data['record']
            hits=[]
            for field in ['reasoning','response']:
                text=record[field]
                for pattern in patterns:
                    for match in re.finditer(pattern,text,re.I|re.S):
                        hits.append(dict(field=field,quote=match.group(),context=text[max(0,match.start()-120):min(len(text),match.end()+250)]))
            if hits:
                flags.append(dict(run_root=str(root),arm=arm,candidate_id=data['candidate_id'],result_path=str(path),result_sha256=runtime.digest(path.read_bytes()),user=record['user'],hits=hits))
report=dict(effective_accepted_counts=dict(counts),patterns=patterns,rows=flags)
target=base/'output/2026-09-15_dataset_refresh_quality_screen/editing_leak_final_snapshot.json'
runtime.save_checkpoint(target,report)
print(json.dumps(dict(counts=dict(counts),flagged=len(flags),hits=sum(len(x['hits']) for x in flags))))
for row in flags:
    print(json.dumps({k:v for k,v in row.items() if k not in ['user','result_path','result_sha256','run_root']},ensure_ascii=False))
