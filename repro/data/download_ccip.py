"""下载 CCIP 训练/评测数据 (deepghs/dinov2_ccip_eval 因 deepghs 私有库存储超限 403, 用公开数据替代)"""
from huggingface_hub import hf_hub_download, snapshot_download

# 训练: character_similarity v1 (zerochan, 4001 角色 / 292800 张), 推测即原 /nfs/ccip/ccip_v1
print(hf_hub_download('deepghs/character_similarity', 'images_v1.tar.xz', repo_type='dataset', local_dir='zips'))
# 评测: 原 cfgs/py/eval/ccip.py 用的 zerochan-character-w640-ws-m50-full 的 test split
print(snapshot_download('animetimm/zerochan-character-w640-ws-m50-full', repo_type='dataset',
                        allow_patterns=['test/*', '*.json', '*.csv', 'README.md']))
