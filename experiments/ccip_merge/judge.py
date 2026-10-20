"""
合并 CCIP 训练集第 2 步: 用服务器上的 codex cli (gpt-6-astra) 判断候选对是否同一角色
python experiments/ccip_merge/judge.py [--limit-batches N] [--jobs 6]
结果按批缓存在 data/ccip_merge/judge/batch_XXXXX.json, 可断点续跑
"""
import argparse
import collections
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

R = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
OUT = R / 'data/ccip_merge'
MODEL = 'gpt-6-astra'

PROMPT = '''You are cleaning an anime character re-identification dataset (used to train CCIP, a model that decides whether
two illustrations show the same character). Three sources were crawled from zerochan.net and use slightly different naming:
  - zc800k: zerochan primary tag, e.g. "Rem (Re:Zero)"; context = the qualifier in parentheses
  - v1:     lowercase folder name; context = game folder (arknights/azurlane/bluearchive/fgo/genshin/girlsfrontline/neuralcloud/others)
  - v2:     zerochan tag; context = series folder (and parent character folder if nested, i.e. the class is a sub-tag of its parent)
For each candidate pair below decide:
  "same"      - the same character identity and the same design lineage, only the name differs (spelling, romanization,
                name order, alias, qualifier, duplicate folder). These will be merged into one class.
  "variant"   - same underlying person but a separately tagged version that usually looks clearly different
                (Alter / Lily / child or adult version, gender-swap, different incarnation in another work, mecha/beast form...).
  "different" - different characters (namesakes from different works, siblings, multi-character pictures, etc.).
  "unsure"    - not enough information.
"shared" = number of identical zerochan images in both classes (n = class sizes). Sharing images is strong evidence of
"same" when the ratio is high, but v2 contains per-person crops of multi-character pictures, so a few shared images
between two different characters of the same series is normal.
Use your knowledge of anime/game characters. Answer every id exactly once. Keep "reason" under 15 words.

Pairs (id | A | B | shared):
'''

SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['results'],
    'properties': {'results': {'type': 'array', 'items': {
        'type': 'object', 'additionalProperties': False, 'required': ['id', 'verdict', 'reason'],
        'properties': {'id': {'type': 'integer'}, 'verdict': {'enum': ['same', 'variant', 'different', 'unsure']},
                       'reason': {'type': 'string'}}}}},
}


def describe(c):
    ctx = f", {c['context']}" if c['context'] else ''
    return f"{c['name']} [{c['source']}{ctx}, n={c['n']}]"


def components(cands):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for p in cands:
        parent[find(p['a'])] = find(p['b'])
    groups = collections.defaultdict(list)
    for i, p in enumerate(cands):
        groups[find(p['a'])].append(i)
    return list(groups.values())


def make_batches(cands, size):
    batches, cur = [], []
    for comp in sorted(components(cands), key=lambda g: min(g)):  # 同一连通块尽量放在同一批, 判断一致
        if cur and len(cur) + len(comp) > size:
            batches.append(cur)
            cur = []
        cur.extend(comp)
    if cur:
        batches.append(cur)
    return batches


def run_batch(bi, idx, cands, classes, workdir):
    out = OUT / 'judge' / f'batch_{bi:05d}.json'
    if out.exists():
        return bi, 'cached'
    lines = []
    for i in idx:
        p = cands[i]
        a, b = classes[p['a']], classes[p['b']]
        lines.append(f"{i} | {describe(a)} | {describe(b)} | shared={p['shared']}")
    prompt = PROMPT + '\n'.join(lines)
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        schema, last = Path(td) / 'schema.json', Path(td) / 'last.json'
        schema.write_text(json.dumps(SCHEMA))
        for attempt in range(3):
            r = subprocess.run(['codex', 'exec', '-m', MODEL, '--skip-git-repo-check', '--ephemeral', '-s', 'read-only',
                                '-C', td, '--output-schema', str(schema), '-o', str(last), '-'],
                               input=prompt, text=True, capture_output=True, timeout=1800)
            try:
                res = json.loads(last.read_text())['results']
                got = {x['id'] for x in res}
                missing = set(idx) - got
                if len(missing) <= len(idx) * 0.02:  # 偶尔漏几个, 剩下的下一轮补
                    res = [x for x in res if x['id'] in set(idx)]
                    out.write_text(json.dumps(dict(ids=idx, results=res, missing=sorted(missing)), ensure_ascii=False))
                    return bi, f'ok {len(res)}/{len(idx)}'
            except Exception as e:  # noqa: BLE001
                err = f'{type(e).__name__}: {e} | {r.stderr[-300:]}'
        return bi, f'FAILED {err}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=150)
    parser.add_argument('--limit-batches', type=int, default=None)
    parser.add_argument('--only', type=str, default=None, help='只跑这些 id (json 文件), 用于补漏')
    args = parser.parse_args()
    classes = json.load(open(OUT / 'classes.json'))
    cands = json.load(open(OUT / 'candidates.json'))
    (OUT / 'judge').mkdir(exist_ok=True)
    workdir = OUT / 'judge_tmp'
    workdir.mkdir(exist_ok=True)
    if args.only:
        only = json.load(open(args.only))
        batches = [only[i:i + args.batch_size] for i in range(0, len(only), args.batch_size)]
        start = 90000
    else:
        batches = make_batches(cands, args.batch_size)
        start = 0
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    print(f'{len(cands)} pairs, {len(batches)} batches', flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_batch, start + bi, idx, cands, classes, workdir) for bi, idx in enumerate(batches)]
        for n, fut in enumerate(as_completed(futs), 1):
            bi, status = fut.result()
            print(f'[{n}/{len(batches)}] batch {bi}: {status}', flush=True)


if __name__ == '__main__':
    main()
