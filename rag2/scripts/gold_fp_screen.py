# -*- coding: utf-8 -*-
"""给 gold_eval 500 条 (query, gold_clause) 用 CE 打分，筛假阳性候选"""
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

clauses = {}
for l in open('data/corpus/clauses.jsonl', encoding='utf-8'):
    if l.strip():
        c = json.loads(l)
        clauses[c['id']] = c['text']
rows = [json.loads(l) for l in open('data/phase7/gold_eval.jsonl', encoding='utf-8') if l.strip()]

tok = AutoTokenizer.from_pretrained('data/models/cross_light')
model = AutoModelForSequenceClassification.from_pretrained('data/models/cross_light').to('cuda')
model.eval()
out = []
with torch.no_grad():
    for i, r in enumerate(rows, 1):
        q = r['query']
        g = clauses[r['clause_id']]
        enc = tok(q, g, padding=True, truncation=True, max_length=128, return_tensors='pt').to('cuda')
        s = float(model(**enc).logits.squeeze(-1))
        out.append({'qid': r['query'][:20], 'clause': r['clause_id'], 'score': round(s, 3),
                    'query': q, 'gold': g[:80]})
        if i % 100 == 0:
            print(f'打分 {i}/500', flush=True)

out.sort(key=lambda x: x['score'])
with open('data/phase7/gold_score.jsonl', 'w', encoding='utf-8') as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + '\n')
scores = [o['score'] for o in out]
ss = sorted(scores)
print('=== 500 条 gold (query,clause) CE 打分 ===')
print(f'min {ss[0]:.3f} / p10 {ss[50]:.3f} / p25 {ss[125]:.3f} / p50 {ss[250]:.3f} / max {ss[-1]:.3f}')
print(f'score<0（高危假阳性候选）: {sum(1 for s in scores if s < 0)}')
print(f'score<1.5（中危）: {sum(1 for s in scores if s < 1.5)}')
print('=== 最低 15 条 ===')
for o in out[:15]:
    print(f'  {o["score"]:+.2f} | {o["qid"]} | {o["clause"][:30]}')
    print(f'     Q: {o["query"][:50]}')
    print(f'     G: {o["gold"][:50]}')
