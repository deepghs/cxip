import torch
from rainbowneko.infer import WorkflowRunner, LoadImageAction, ForwardAction, VisPredAction, BuildModelAction, \
    PrepareAction, LoadModelAction, BasicAction, feedback_input
from torchvision import transforms
from rainbowneko.models.wrapper import FeatWrapper
from model import CAFormerBackbone
from functools import partial
from einops import repeat
import math

from PIL import Image
import random
class CropAndStitch:
    def __init__(self, img_size=384, n=12):
        assert img_size % n == 0
        self.n = n
        self.m = img_size // n

    def __call__(self, image):
        width, height = image.size

        # 创建一个新的空白图像,用于拼接裁剪后的图像块
        stitched_image = Image.new('RGB', (self.n * self.m, self.n * self.m))

        # 随机裁剪图像块并拼接
        for i in range(self.n):
            for j in range(self.n):
                # 随机生成裁剪位置的左上角坐标
                left = random.randint(0, width - self.m)
                top = random.randint(0, height - self.m)

                # 裁剪图像块
                patch = image.crop((left, top, left + self.m, top + self.m))

                # 将裁剪后的图像块拼接到新图像上
                stitched_image.paste(patch, (i * self.m, j * self.m))

        return stitched_image

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(640),
    transforms.CenterCrop(384),
    CropAndStitch(384, 12),
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
    #BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_m36'))),
    #LoadModelAction({'model': 'exps/csip_v1_info_nce_m36-p384-weak/ckpts/csip-caformer-m36-12000.ckpt'}),
    BuildModelAction(partial(FeatWrapper, model=CAFormerBackbone('caformer_s18'))),
    LoadModelAction({'model': 'exps/csip_v1_noisy_info_nce_s18-p384-patch/ckpts/csip-caformer-s18-12000.ckpt'}),

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