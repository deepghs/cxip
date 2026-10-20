"""
把 ccip_v1 对齐到原始 /nfs/ccip/ccip_v1 (hub.deepghs.org deepghs/ccip 的 ccip.tar 备份):
原始集合是 HF character_similarity images_v1 的子集 (清洗过), 多出来的图片挪到 ccip_v1_extra/, 并重建 ccip_v1_flat。
只做 rename / symlink, 不解压图片。
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("REPRO_ROOT", "/data/neko/cxip_repro")) / "data"
orig = set()
for l in open(DATA / "zips/khub_ccip/ccip.tar.list"):
    l = l.rstrip("\n")
    if l.startswith("ccip/ccip_v1/") and not l.endswith("/"):
        orig.add(l[len("ccip/ccip_v1/"):])

moved = 0
for root, _, fs in os.walk(DATA / "ccip_v1"):
    for f in fs:
        rel = os.path.relpath(os.path.join(root, f), DATA / "ccip_v1")
        if rel not in orig:
            dst = DATA / "ccip_v1_extra" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(DATA / "ccip_v1" / rel, dst)
            moved += 1
print("moved", moved)

flat = DATA / "ccip_v1_flat"
n = 0
for link in sorted(flat.iterdir()):
    target = link.resolve()
    if not target.is_dir():  # 上次运行留下的悬空链接
        link.unlink()
    elif not any(target.iterdir()):  # 原始数据里没有的角色
        link.unlink()
        target.rmdir()
    else:
        n += 1
print("ccip_v1_flat chars", n)
