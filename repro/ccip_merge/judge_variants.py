"""
合并 CCIP 训练集: 对 judge.py 判为 "variant" 的类别对细分, 决定哪些也合并
规则 (2026-10-07 用户确认): Alter / 泳装·换职阶等 "一眼能看出是同一个角色" 的变体合并; 性转不合并。
codex (gpt-6-astra) 给出类别和 merge 决定, 并参考 visual_check.py 的跨类同角色概率。
输出 data/ccip_merge/variant_judge.json
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

R = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
SRC = R / 'data/ccip_merge'
MODEL = 'gpt-6-astra'
PROMPT = '''These pairs of classes from an anime character re-identification dataset (zerochan tags) were judged to be the
same underlying person but separately tagged versions. Decide for each pair whether they should be MERGED into one
identity class for training a "same character?" model. The rule: merge if a viewer would readily recognize both as the
same character (same face, hair and core design), even if outfit or class differs; do not merge if the design changes
so much that they look like a different character.
Categories:
  alter_or_dark      - Alter / dark / corrupted / awakened version (MERGE)
  outfit_or_class    - swimsuit, seasonal, festival, costume, different Servant class or skin of the same character (MERGE)
  age_variant        - child / adult / older version (MERGE only if still clearly recognizable)
  other_incarnation  - same character in another work / reincarnation / alternate universe (MERGE only if the design is clearly the same)
  gender_swap        - gender-bent version (NEVER merge)
  form_change        - beast / mecha / monster / non-human form, fusion (do not merge)
  not_variant        - actually unrelated characters or an identical duplicate (merge only if identical duplicate)
"visual" = mean same-character probability from a CCIP model between sample images of the two classes (0..1; same
character usually > 0.7, different characters usually < 0.1). Use it as supporting evidence together with your knowledge.
Answer every id exactly once; "reason" under 15 words.

Pairs (id | A | B | shared images | visual):
'''
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['results'],
          'properties': {'results': {'type': 'array', 'items': {
              'type': 'object', 'additionalProperties': False, 'required': ['id', 'category', 'merge', 'reason'],
              'properties': {'id': {'type': 'integer'},
                             'category': {'enum': ['alter_or_dark', 'outfit_or_class', 'age_variant', 'other_incarnation',
                                                   'gender_swap', 'form_change', 'not_variant']},
                             'merge': {'type': 'boolean'}, 'reason': {'type': 'string'}}}}}}


def main():
    classes = json.load(open(SRC / 'classes.json'))
    cands = json.load(open(SRC / 'candidates.json'))
    vc = pd.read_parquet(SRC / 'visual_check.parquet').set_index('id')
    ids = [int(i) for i in vc.index[vc['verdict'] == 'variant']]

    def desc(c):
        return f"{c['name']} [{c['source']}{', ' + c['context'] if c['context'] else ''}, n={c['n']}]"
    lines = [f"{i} | {desc(classes[cands[i]['a']])} | {desc(classes[cands[i]['b']])} | shared={cands[i]['shared']} | "
             f"visual={vc.loc[i, 'cross']:.2f}" for i in ids]
    results = {}
    for start in range(0, len(lines), 100):
        chunk = lines[start:start + 100]
        want = set(ids[start:start + 100])
        with tempfile.TemporaryDirectory() as td:
            schema, last = Path(td) / 'schema.json', Path(td) / 'last.json'
            schema.write_text(json.dumps(SCHEMA))
            for _ in range(3):
                subprocess.run(['codex', 'exec', '-m', MODEL, '--skip-git-repo-check', '--ephemeral', '-s', 'read-only',
                                '-C', td, '--output-schema', str(schema), '-o', str(last), '-'],
                               input=PROMPT + '\n'.join(chunk), text=True, capture_output=True, timeout=1800)
                try:
                    got = {x['id']: x for x in json.loads(last.read_text())['results'] if x['id'] in want}
                except Exception:  # noqa: BLE001
                    continue
                results.update(got)
                if want <= set(got):
                    break
        print(start, len(results), flush=True)
    for i, x in results.items():
        if x['category'] == 'gender_swap':  # 用户规则: 性转一律不合并
            x['merge'] = False
        x.update(a=classes[cands[i]['a']]['key'], b=classes[cands[i]['b']]['key'], visual=float(vc.loc[i, 'cross']))
    missing = sorted(set(ids) - set(results))
    json.dump(dict(results=list(results.values()), missing=missing), open(SRC / 'variant_judge.json', 'w'), ensure_ascii=False, indent=1)
    df = pd.DataFrame(results.values())
    print(df.groupby('category')['merge'].agg(['size', 'sum']))
    print('missing', missing)


if __name__ == '__main__':
    main()
