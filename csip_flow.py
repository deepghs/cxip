import torch
from rainbowneko.infer import WorkflowRunner, LoadImageAction, ForwardAction, VisPredAction, BuildModelAction, PrepareAction, LoadModelAction
from torchvision import transforms
from rainbowneko.models.wrapper import FeatWrapper
from model import CAFormerBackbone
from functools import partial

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


actions=[
    PrepareAction(device='cuda', dtype=torch.float32),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36', input_resolution=384))),
    LoadModelAction({'model': 'ckpts/csip-caformer-m36-7800.ckpt'}),

    LoadImageAction(
        image_paths=[
            'imgs/113389072_p0_master1200.jpg',
            'imgs/113901432_p0_master1200.jpg',
            'imgs/115706741_p0_master1200.jpg',
            'imgs/1.jpg',
            'imgs/2.jpg'
        ],
        image_transforms=EVAL_TRANSFORM,
    ),
    ForwardAction(),
    VisPredAction(),
]

if __name__ == '__main__':
    runner = WorkflowRunner()
    runner.run(actions)