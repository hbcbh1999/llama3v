from .config import Config
from .projection import Projection
import torch
from PIL import Image
from transformers import LlamaTokenizer, LlamaPreTrainedModel, LlamaForCausalLM, AutoModel, PreTrainedTokenizerFast, TextIteratorStreamer
from vision_embedding import get_vision_embedding

# TODO: We are cleaning this code up right now and going to push as soon as it's done.


class Llama3vPreTrainedModel(LlamaPreTrainedModel):
    config_class = Config


class Llama3v(Llama3vPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaForCausalLM(config)
        self.vision_model = get_vision_embedding

    def project_model(self, embed_dim, vision_dim):
        pass
