"""把 AnimeCCIP-v3 / AnimeCSIP-v3 复制一份到 huggingface.co 的 deepghs-cv (private), 并各建一个 ZeroGPU demo Space (private)。
Space 放在个人账号 SPACE_OWNER 下: 2026-07 起组织建 Gradio Space (含 cpu-basic / ZeroGPU) 需要 Team/Enterprise,
免费个人账号可以有 2 个 ZeroGPU Gradio Space; 创建时直接指定 zero-a10g (先建 cpu-basic 会 402)。

模型文件来自 /data/neko/cxip_v3/release/<name>/, README 用 release/hf_copy/<name>.README.md (repo id 改为 deepghs-cv,
去掉 HF_ENDPOINT, 加 Space 链接)。Space 文件来自 release/spaces/<name>/ (app.py 源码在 v3/release_space/)。
token: 环境变量 HFCO_TOKEN (source ~/.config/cxip/env.sh), 同时写进 Space secret HF_TOKEN 以读取私有模型。
用法: env -u HF_ENDPOINT python -m v3.scripts.publish_hf_cv {check,models,readme,spaces,test} [--only AnimeCCIP-v3]
"""
import argparse
import hashlib
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REL = Path('/data/neko/cxip_v3/release')
ORG = 'deepghs-cv'
SPACE_OWNER = '7eu7d7'
NAMES = ['AnimeCCIP-v3', 'AnimeCSIP-v3']
EP = 'https://huggingface.co'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def remote_sha(api, rid, info, f, repo_type):
    s = info[f]
    if s.lfs:
        return s.lfs.sha256
    p = api.hf_hub_download(rid, f, repo_type=repo_type, force_download=True, cache_dir='/tmp/hfco_verify')
    return sha256(p)


def verify(api, rid, files, repo_type='model'):
    """files: {remote path: local path}; 返回不一致的文件列表。"""
    meta = api.repo_info(rid, repo_type=repo_type, files_metadata=True)
    info = {s.rfilename: s for s in meta.siblings}
    bad = [f for f, lp in files.items() if f not in info or remote_sha(api, rid, info, f, repo_type) != sha256(lp)]
    print(f'  verify {rid}: {len(files) - len(bad)}/{len(files)} sha256 OK, private={meta.private}', flush=True)
    return bad


def check(api):
    w = api.whoami()
    orgs = {o['name']: o.get('roleInOrg') for o in w.get('orgs', [])}
    print('user', w['name'], '| token role', w.get('auth', {}).get('accessToken', {}).get('role'),
          '| deepghs-cv role', orgs.get(ORG))
    print('base_model deepghs/AnimeDINOv3-timm exists on HF:', api.repo_exists('deepghs/AnimeDINOv3-timm'))
    assert ORG in orgs, f'token has no access to {ORG}'


def build_readme(api, name):
    readme = (REL / 'hf_copy' / f'{name}.README.md').read_text()
    readme = readme.replace(f'spaces/{ORG}/{name}', f'spaces/{SPACE_OWNER}/{name}').replace(
        f'demo Space: [{ORG}/{name}]', f'demo Space: [{SPACE_OWNER}/{name}]')
    if not api.repo_exists('deepghs/AnimeDINOv3-timm'):   # HF 上没有该 repo 时去掉 base_model 元数据 (正文链接保留)
        readme = readme.replace('base_model: deepghs/AnimeDINOv3-timm\n', '')
    rd = REL / 'hf_copy' / f'{name}.README.upload.md'
    rd.write_text(readme)
    return rd


def readme_only(api, names):
    for name in names:
        rid = f'{ORG}/{name}'
        rd = build_readme(api, name)
        c = api.create_commit(rid, [CommitOperationAdd('README.md', str(rd))], commit_message='README: demo Space link')
        print(rid, c.oid[:10], 'README', flush=True)
        assert not verify(api, rid, {'README.md': rd})


def models(api, names):
    for name in names:
        rid = f'{ORG}/{name}'
        rd = build_readme(api, name)
        files = {str(p.relative_to(REL / name)): p for p in sorted((REL / name).rglob('*'))
                 if p.is_file() and p.name != 'README.md'}
        files['README.md'] = rd
        api.create_repo(rid, private=True, exist_ok=True)
        c = api.create_commit(rid, [CommitOperationAdd(f, str(p)) for f, p in files.items()],
                              commit_message='Upload AnimeCxIP-v3 (copy of hub.deepghs.org release)')
        print(rid, c.oid[:10], f'{len(files)} files', flush=True)
        assert not verify(api, rid, files)


def spaces(api, names, token):
    for name in names:
        rid, model = f'{SPACE_OWNER}/{name}', f'{ORG}/{name}'
        d = REL / 'spaces' / name
        api.create_repo(rid, repo_type='space', space_sdk='gradio', private=True, exist_ok=True,
                        space_hardware='zero-a10g',
                        space_secrets=[{'key': 'HF_TOKEN', 'value': token}],
                        space_variables=[{'key': 'MODEL_REPO', 'value': model}])
        files = {str(p.relative_to(d)): p for p in sorted(d.rglob('*')) if p.is_file()}
        c = api.create_commit(rid, [CommitOperationAdd(f, str(p)) for f, p in files.items()], repo_type='space',
                              commit_message='Gradio demo (ZeroGPU)')
        print(rid, 'space', c.oid[:10], f'{len(files)} files', flush=True)
        assert not verify(api, rid, files, 'space')
        rt = api.get_space_runtime(rid)
        print('  hardware', rt.hardware, '| requested', rt.requested_hardware, '| stage', rt.stage, flush=True)


def wait_and_test(api, names, token, timeout=1800):
    from gradio_client import Client, handle_file
    ex = REL / 'spaces' / 'AnimeCCIP-v3' / 'examples'
    for name in names:
        rid = f'{SPACE_OWNER}/{name}'
        t0, last = time.time(), None
        while time.time() - t0 < timeout:
            rt = api.get_space_runtime(rid)
            if rt.stage != last:
                print(f'  {rid} [{time.time() - t0:5.0f}s] stage={rt.stage} hardware={rt.hardware}', flush=True)
                last = rt.stage
            if rt.stage in ('RUNNING', 'BUILD_ERROR', 'RUNTIME_ERROR', 'CONFIG_ERROR', 'NO_APP_FILE', 'PAUSED'):
                break
            time.sleep(20)
        if last != 'RUNNING':
            print(f'  {rid}: not running ({last}); see the Space logs')
            continue
        c = Client(rid, hf_token=token, verbose=False)
        thr = -0.5819 if 'CSIP' in name else 0.3358
        for b in ('artistB_koishi.jpg', 'artistC_reimu.jpg'):
            t = time.time()
            v, det = c.predict(handle_file(str(ex / 'reference.jpg')), handle_file(str(ex / b)), thr,
                               api_name='/compare')
            print(f'  {rid} compare reference vs {b}: {v.strip("# ")} | {det.splitlines()[0]} | {time.time() - t:.1f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['check', 'models', 'readme', 'spaces', 'test'])
    ap.add_argument('--only', choices=NAMES)
    args = ap.parse_args()
    token = os.environ['HFCO_TOKEN']
    api = HfApi(endpoint=EP, token=token)
    names = [args.only] if args.only else NAMES
    if args.step == 'check':
        check(api)
    elif args.step == 'models':
        models(api, names)
    elif args.step == 'readme':
        readme_only(api, names)
    elif args.step == 'spaces':
        spaces(api, names, token)
    else:
        wait_and_test(api, names, token)


if __name__ == '__main__':
    main()
