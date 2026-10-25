"""去偏 / 续训实验汇总: crop384 套件 (PR-AUC + 偏置 + 同角色负对误报率) 与 native_full。
python -m v3.scripts.summarize_dft [root=/data/neko/cxip_v3/exps/csip_p2] [prefix=dft_]"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else '/data/neko/cxip_v3/exps/csip_p2')
prefix = sys.argv[2] if len(sys.argv) > 2 else 'dft_'
rows = [('teacher_v1 (EMA crop / 非EMA native)', Path('/data/neko/cxip_v3/exps/final/csip_teacher_v1'), 'eval_native_last_raw')]
rows += [(d.name, d, 'eval_native') for d in sorted(root.glob(prefix + '*'))]
print('| 实验 | crop 0918 | crop public | I_content | I_char | 同角色误报 | native 0918 | native public |')
print('|-|-|-|-|-|-|-|-|')
f = lambda v, n=3: '' if v is None else f'{v:.{n}f}'  # noqa: E731
for name, d, nat in rows:
    c = json.loads((d / 'eval/suite.json').read_text()) if (d / 'eval/suite.json').exists() else {}
    n = json.loads((d / nat / 'suite.json').read_text()) if (d / nat / 'suite.json').exists() else {}
    cd, nd, b = c.get('datasets', {}), n.get('datasets', {}), c.get('bias', {})
    hn = (cd.get('test100_0918', {}).get('crosschar') or {}).get('hardneg_fpr')
    print(f'| {name} | {f(cd.get("test100_0918", {}).get("pr_auc"))} | {f(cd.get("csip_eval_public", {}).get("pr_auc"))} | '
          f'{f(b.get("content", {}).get("I"))} | {f(b.get("char", {}).get("I"))} | {f(hn, 4)} | '
          f'{f(nd.get("test100_0918", {}).get("pr_auc"))} | {f(nd.get("csip_eval_public", {}).get("pr_auc"))} |')
