from transformers import LlamaConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionConfig


class Config(LlamaConfig):
    def __init__(self, **kwargs):
        self.scale_factor = 1.5
        self.use_cache = True
        self.query_num = 64
        self.image_size = 448

        super().__init__(**kwargs)
