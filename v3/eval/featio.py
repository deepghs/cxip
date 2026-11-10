"""特征 / 元数据写入的保护 (extract_release_feats, gap_feats, eval_kaloscope 共用).

- 目标是符号链接: 无论是否 --overwrite 都拒绝 (不经链接写到别处的数据上, 如 v3_x4/feats 里指向旧目录的 legacy npz).
- 目标已存在: 默认拒绝, --overwrite 才覆盖.
- meta.json: 内容相同则不重写 (幂等); 不同且没有 --overwrite 时拒绝.
提特征的脚本应在加载模型之前对全部目标调用 check_targets, 尽早失败.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RefuseWrite(RuntimeError):
    pass


def check_target(path, overwrite: bool = False) -> Path:
    p = Path(path)
    if p.is_symlink():
        raise RefuseWrite(f'refusing to write through symlink {p} -> {p.readlink()} (even with --overwrite); '
                          'remove the link or pick another output dir')
    if p.exists() and not overwrite:
        raise RefuseWrite(f'{p} exists; pass --overwrite to replace it')
    return p


def check_targets(paths, overwrite: bool = False) -> None:
    for p in paths:
        check_target(p, overwrite)


def preflight(npz_paths, meta_path=None, overwrite: bool = False) -> None:
    """加载模型之前: 全部 npz 目标 check_target; meta 只查符号链接 (内容是否相同要等模型加载后才知道)"""
    if meta_path is not None and Path(meta_path).is_symlink():
        check_target(meta_path, overwrite)
    check_targets(npz_paths, overwrite)


def save_feats(path, feats, sample_key=None, overwrite: bool = False) -> None:
    """runner 的 npz 格式: feats (fp16), 可选 sample_key (str)"""
    p = check_target(path, overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(feats=np.asarray(feats).astype(np.float16))
    if sample_key is not None:
        kw['sample_key'] = np.asarray(sample_key).astype(str)
    with open(p, 'wb') as fh:                   # 显式文件对象: np.savez 不会再给路径补 .npz
        np.savez(fh, **kw)


def write_json(path, obj, overwrite: bool = False) -> bool:
    """写 JSON; 已存在且内容相同 -> 不写, 返回 False; 不同 -> 需要 overwrite. 返回是否写了"""
    p = Path(path)
    if p.is_symlink():
        check_target(p, overwrite)
    text = json.dumps(obj, indent=1, default=float)
    if p.exists():
        try:
            same = json.loads(p.read_text()) == json.loads(text)
        except ValueError:
            same = False
        if same:
            return False
        if not overwrite:
            raise RefuseWrite(f'{p} exists with different content; pass --overwrite to replace it')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return True
