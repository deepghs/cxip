from transformers import PretrainedConfig

class CCIPConfig(PretrainedConfig):
    model_type = "ccip"

    def __init__(
        self,
        backbone="caformer_b36",
        input_resolution=384,
        heads=8,
        out_dims=768,
        **kwargs,
    ):
        self.backbone = backbone
        self.input_resolution = input_resolution
        self.heads = heads
        self.out_dims = out_dims
        super().__init__(**kwargs)

