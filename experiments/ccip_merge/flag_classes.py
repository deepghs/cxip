"""
合并 CCIP 训练集: 让 codex (gpt-6-astra) 扫一遍合并后的全部类别, 找出不是"单个具体角色"的类别
(种族/物种群体/泛称/原创角色合集/多人组合/非角色等), 这些会给 CCIP 引入错误的正样本
python experiments/ccip_merge/flag_classes.py --jobs 6   -> data/ccip_merged/flags/batch_*.json
"""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

R = Path(os.environ.get('REPRO_ROOT', '/data/neko/cxip_repro'))
OUT = R / 'data/ccip_merged'
MODEL = 'gpt-6-astra'
PROMPT = '''Below are class names of an anime character re-identification dataset (crawled from zerochan.net; each class should
contain images of ONE specific character so that any two images of a class are a "same character" positive pair).
Flag every class that is NOT a single specific character identity, e.g.
  - races / species groups / factions whose members look different (e.g. "Miqo'te", "Elf", "Kemonomimi")
  - generic or collective tags ("Fan Character", "Original", "OC", "Gijinka", "Mascot" collections, "Genderbend" groups)
  - classes that are explicitly several characters (duos, groups, "X & Y", a whole cast)
  - non-characters (objects, places, vehicles, memes, real people / cosplayers, voice actors)
Do NOT flag: normal named characters (including named Pokémon/monster species with a fixed design, ship girls, servants,
vocaloids, vtubers, mascot characters with a fixed design), or a specific named variant of a character.
Return only the flagged ids (empty list if none). Keep "reason" under 12 words.

Classes (id | name | member tags from the source datasets):
'''
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['flagged'],
          'properties': {'flagged': {'type': 'array', 'items': {
              'type': 'object', 'additionalProperties': False, 'required': ['id', 'category', 'reason'],
              'properties': {'id': {'type': 'integer'},
                             'category': {'enum': ['group_or_species', 'generic_or_collective', 'multiple_characters',
                                                   'non_character', 'uncertain']},
                             'reason': {'type': 'string'}}}}}}


def run_batch(bi, items, workdir):
    out = OUT / 'flags' / f'batch_{bi:05d}.json'
    if out.exists():
        return bi, 'cached'
    lines = [f"{c['label']} | {c['name']} | " + '; '.join(m['key'] for m in c['members'][:4]) for c in items]
    ids = {c['label'] for c in items}
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        schema, last = Path(td) / 'schema.json', Path(td) / 'last.json'
        schema.write_text(json.dumps(SCHEMA))
        err = ''
        for _ in range(3):
            r = subprocess.run(['codex', 'exec', '-m', MODEL, '--skip-git-repo-check', '--ephemeral', '-s', 'read-only',
                                '-C', td, '--output-schema', str(schema), '-o', str(last), '-'],
                               input=PROMPT + '\n'.join(lines), text=True, capture_output=True, timeout=1800)
            try:
                flagged = [x for x in json.loads(last.read_text())['flagged'] if x['id'] in ids]
                out.write_text(json.dumps(dict(ids=sorted(ids), flagged=flagged), ensure_ascii=False))
                return bi, f'ok {len(flagged)} flagged'
            except Exception as e:  # noqa: BLE001
                err = f'{type(e).__name__}: {e} | {r.stderr[-300:]}'
        return bi, f'FAILED {err}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=300)
    args = parser.parse_args()
    classes = json.load(open(OUT / 'classes.json'))
    (OUT / 'flags').mkdir(exist_ok=True)
    workdir = OUT / 'flags_tmp'
    workdir.mkdir(exist_ok=True)
    batches = [classes[i:i + args.batch_size] for i in range(0, len(classes), args.batch_size)]
    print(len(classes), 'classes', len(batches), 'batches', flush=True)
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(run_batch, bi, b, workdir) for bi, b in enumerate(batches)]
        for n, fut in enumerate(as_completed(futs), 1):
            bi, status = fut.result()
            print(f'[{n}/{len(batches)}] batch {bi}: {status}', flush=True)


if __name__ == '__main__':
    main()
