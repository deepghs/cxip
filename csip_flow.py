import torch
from rainbowneko.infer import WorkflowRunner, LoadImageAction, ForwardAction, VisPredAction, BuildModelAction, \
    PrepareAction, LoadModelAction, BasicAction, feedback_input
from torchvision import transforms
from rainbowneko.models.wrapper import FeatWrapper
from model import CAFormerBackbone
from functools import partial
from einops import repeat
import math

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

class CatImageAction(BasicAction):
    @feedback_input
    def forward(self, input, **states):
        bs = input['x'].shape[0]
        input['x_ref'] = repeat(input['x'], 'b c h w -> (n b) c h w', n=bs)
        input['x'] = repeat(input['x'], 'b c h w -> (b n) c h w', n=bs)
        return {'input': input}
    
class ReshapeAction(BasicAction):
    @feedback_input
    def forward(self, output, **states):
        bs = int(math.sqrt(len(output['pred'])))
        output['pred'] = output['pred'].view(bs, bs)
        return {'output': output}

actions=[
    PrepareAction(device='cuda', dtype=torch.float32),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36'))),
    LoadModelAction({'model': 'exps/csip_v1_info_nce_m36-p384-weak/ckpts/csip-caformer-m36-12000.ckpt'}),

    LoadImageAction(
        image_paths=[
            'imgs/113389072_p0_master1200.jpg',
            'imgs/113901432_p0_master1200.jpg',
            'imgs/115706741_p0_master1200.jpg',
            'imgs/1.jpg',
            'imgs/2.jpg',
            'imgs/3.jpg',
        ],
        image_transforms=EVAL_TRANSFORM,
    ),
    #CatImageAction(),
    ForwardAction(),
    #ReshapeAction(),
    VisPredAction(),
]

if __name__ == '__main__':
    runner = WorkflowRunner()
    runner.run(actions)