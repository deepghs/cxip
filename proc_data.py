from pathlib import Path

src = Path("/nfs/ccip/ccip_v1")
dst = Path("/nfs/ccip/ccip_v1_flat")

for file in src.glob("**/*.jpg"):
    name = file.name
    dst_file = dst / f'{file.parent.parent.name}_{file.parent.name}'
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.symlink_to(file)




