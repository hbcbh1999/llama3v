from transformers import LlamaConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionConfig


class Config(LlamaConfig):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1.5
        self.use_cache = True
        self.query_num = 64
        self.image_size = 448

        self.vision_config = Idefics2VisionConfig(
            hidden_size=int(self.scale_factor *
                            Idefics2VisionConfig().hidden_size),
            image_size=int(self.scale_factor *
                           Idefics2VisionConfig().image_size),
            intermediate_size=int(
                self.scale_factor * Idefics2VisionConfig().intermediate_size),
            num_attention_heads=int(
                self.scale_factor * Idefics2VisionConfig().num_attention_heads),
            num_hidden_layers=int(
                self.scale_factor * Idefics2VisionConfig().num_hidden_layers),
            patch_size=14
        )
