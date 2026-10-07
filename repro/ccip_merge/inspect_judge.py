"""抽查 codex 判断结果: python repro/ccip_merge/inspect_judge.py [每类抽样数]"""
import collections
import glob
import json
import os
import random
import sys
from pathlib import Path

OUT = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro')) / 'data/ccip_merge'
classes = json.load(open(OUT / 'classes.json'))
cands = json.load(open(OUT / 'candidates.json'))


def d(c):
    return f"{c['name']} [{c['source']},{c['context']},n={c['n']}]"


res = [x for f in sorted(glob.glob(str(OUT / 'judge/batch_*.json'))) for x in json.load(open(f))['results']]
print(len(res), collections.Counter(x['verdict'] for x in res))
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
by = collections.defaultdict(list)
for x in res:
    by[x['verdict']].append(x)
for v, xs in by.items():
    print('==', v)
    for x in random.sample(xs, min(int(sys.argv[1]) if len(sys.argv) > 1 else 8, len(xs))):
        p = cands[x['id']]
        print('  ', d(classes[p['a']]), '<>', d(classes[p['b']]), 'shared', p['shared'], '|', x['reason'])
