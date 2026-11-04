# SPDX-License-Identifier: AGPL-3.0-only
"""Gradio demo for AnimeCCIP-v3 / AnimeCSIP-v3 (one app, picked by MODEL_REPO).

Compare two images, or rank query images against a few reference images. Scores are the model's calibrated logits:
`logit > config.threshold` means the same character (CCIP) / the same artist (CSIP), sigmoid(logit) is an approximate
probability.
"""
import copy
import os

import spaces  # must come before torch touches CUDA (ZeroGPU)
import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_REPO = os.environ.get('MODEL_REPO', 'deepghs-cv/AnimeCCIP-v3')
CSIP = 'CSIP' in MODEL_REPO
WHAT, WHATS = ('artist', 'artists') if CSIP else ('character', 'characters')
MAX_REFS, MAX_QUERIES = 16, 32
EX = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')

model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True, dtype=torch.bfloat16).eval()
sim = copy.deepcopy(model.sim).float()      # scoring on CPU (CUDA is only available inside @spaces.GPU on ZeroGPU)
model.to('cuda')
processor = AutoImageProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
THRESHOLD = float(model.config.threshold)


def _rgb(im):
    return im if im.mode in ('RGB', 'RGBA') else im.convert('RGBA' if 'transparency' in im.info else 'RGB')


@spaces.GPU(duration=60)
def embed(images):
    pv = processor([_rgb(im) for im in images])['pixel_values']   # tensor, or a list of mixed sizes (CSIP)
    return model.get_image_features(pv).cpu()


def logits(a, b):
    with torch.inference_mode():
        return sim(a, b)


def compare(a, b, thr):
    if a is None or b is None:
        raise gr.Error('Please upload two images.')
    f = embed([a, b])
    s = float(logits(f[:1], f[1:])[0, 0])
    same = s > thr
    verdict = f'### {"Same " + WHAT if same else "Different " + WHATS}'
    detail = (f'Score **{s:.2f}** {">" if same else "≤"} threshold {thr:.3g}  \n'
              f'Approximate probability of the same {WHAT}: **{torch.sigmoid(torch.tensor(s)).item():.1%}**'
              + (' (rough; calibration depends on the image source)' if CSIP else ''))
    return verdict, detail


def _pil_list(gallery):
    return [x[0] if isinstance(x, (tuple, list)) else x for x in (gallery or [])]


def rank(refs, queries, thr):
    refs, queries = _pil_list(refs), _pil_list(queries)
    if not refs or not queries:
        raise gr.Error('Please upload at least one reference image and one query image.')
    if len(refs) > MAX_REFS or len(queries) > MAX_QUERIES:
        raise gr.Error(f'At most {MAX_REFS} reference images and {MAX_QUERIES} query images.')
    f = embed(refs + queries)
    s = logits(f[len(refs):], f[:len(refs)]).mean(1)       # class score = mean logit over the references
    order = s.argsort(descending=True).tolist()
    out = [(queries[i], f'#{r + 1}  {s[i]:.2f}  {"match" if s[i] > thr else "no match"}') for r, i in enumerate(order)]
    n = int((s > thr).sum())
    return out, f'**{n} / {len(queries)}** query images are predicted to show the same {WHAT} as the references.'


def ex(*names):
    return [os.path.join(EX, n) for n in names]


# 示例: 一张参考图 (Artist B 的 Reimu) + 其余 8 张 (3 角色 x 3 画师)
RANK_QUERIES = [f'artist{x}_{c}.jpg' for x in 'ABC' for c in ('koishi', 'reimu', 'flandre') if (x, c) != ('B', 'reimu')]


NAME = MODEL_REPO.split('/')[-1]
INTRO = f'''# {NAME}

{"**CSIP** embeds an illustration so that works by **the same artist** land close together, whatever they depict."
 if CSIP else
 "**CCIP** embeds an illustration so that images of **the same character** land close together, whoever drew them."}
Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}). Default threshold {THRESHOLD:.3g} (best F1 on the test set);
scores above it mean the same {WHAT}.
'''

with gr.Blocks(title=NAME) as demo:
    gr.Markdown(INTRO)
    with gr.Accordion('Threshold', open=False):
        thr = gr.Slider(-8, 8, value=THRESHOLD, step=0.01, label='Decision threshold (logit)',
                        info=f'Higher = stricter. Default {THRESHOLD:.3g}.')
    with gr.Tab('Compare two images'):
        with gr.Row():
            a = gr.Image(type='pil', label='Image A', height=360)
            b = gr.Image(type='pil', label='Image B', height=360)
        btn = gr.Button('Compare', variant='primary')
        verdict, detail = gr.Markdown(), gr.Markdown()
        btn.click(compare, [a, b, thr], [verdict, detail])
        gr.Examples([ex(*p) for p in (
            [('reference.jpg', 'artistB_koishi.jpg'), ('reference.jpg', 'artistC_reimu.jpg')] if CSIP else
            [('reference.jpg', 'artistC_reimu.jpg'), ('reference.jpg', 'artistB_koishi.jpg')])], [a, b])
    with gr.Tab('Rank against references'):
        gr.Markdown(f'Upload a few images of one {WHAT} as references, then the images to check. Each query is scored '
                    f'by its mean score against the references and ranked.')
        with gr.Row():
            refs = gr.Gallery(type='pil', label=f'References (1-{MAX_REFS} images of one {WHAT})', height=300,
                              columns=4, interactive=True, allow_preview=False, object_fit='contain')
            queries = gr.Gallery(type='pil', label=f'Queries (up to {MAX_QUERIES})', height=300, columns=4,
                                 interactive=True, allow_preview=False, object_fit='contain')
        with gr.Row():
            load = gr.Button('Load example images')
            btn2 = gr.Button('Rank', variant='primary')
        summary = gr.Markdown()
        ranked = gr.Gallery(label='Queries, best match first', columns=4, height=640, object_fit='contain')
        btn2.click(rank, [refs, queries, thr], [ranked, summary])
        load.click(lambda: (ex('reference.jpg'), ex(*RANK_QUERIES)), None, [refs, queries])
    gr.Markdown('Example images come from the held-out test set and belong to their artists; artist names are '
                'withheld.')

if __name__ == '__main__':
    demo.launch()
