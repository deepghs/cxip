# SPDX-License-Identifier: AGPL-3.0-only
from transformers import PretrainedConfig


class CxIPConfig(PretrainedConfig):
    """CxIP v3 (CCIP: character identity / CSIP: artist style) on an AnimeDINOv3-2B backbone."""

    model_type = "cxip"

    def __init__(
        self,
        task="ccip",
        image_size=512,
        patch_size=16,
        embed_dim=2688,
        depth=24,
        num_heads=21,
        mlp_ratio=16 / 3,
        num_queries=16,
        head_dim=1024,
        head_num_heads=16,
        head_hidden=2048,
        out_dim=768,
        matryoshka_dims=(768, 512, 256, 128),
        logit_scale_init=2.302585092994046,
        logit_bias_init=-10.0,
        threshold=0.0,
        **kwargs,
    ):
        self.task = task
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_queries = num_queries
        self.head_dim = head_dim
        self.head_num_heads = head_num_heads
        self.head_hidden = head_hidden
        self.out_dim = out_dim
        self.matryoshka_dims = list(matryoshka_dims)
        self.logit_scale_init = logit_scale_init
        self.logit_bias_init = logit_bias_init
        self.threshold = threshold
        super().__init__(**kwargs)
