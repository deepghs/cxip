"""查看某些类别名相关的所有候选对及 codex 判定: python experiments/ccip_merge/show_group.py Nami Suzuya"""
import glob
import json
import os
import sys
from pathlib import Path

OUT = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data/ccip_merge'
classes = json.load(open(OUT / 'classes.json'))
cands = json.load(open(OUT / 'candidates.json'))
res = {x['id']: x for f in glob.glob(str(OUT / 'judge/batch_*.json')) for x in json.load(open(f))['results']}
names = {n.lower() for n in sys.argv[1:]}
for i, p in enumerate(cands):
    a, b = classes[p['a']], classes[p['b']]
    base = lambda c: c['name'].split(' (')[0].lower()  # noqa: E731
    if base(a) in names or base(b) in names:
        r = res.get(i, {})
        print(f"{r.get('verdict', '?'):9s} {a['source']}:{a['key']} (n={a['n']}) <> {b['source']}:{b['key']} (n={b['n']}) "
              f"shared={p['shared']} | {r.get('reason', '')}")
