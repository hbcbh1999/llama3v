from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer
import torch


def vision_model():
    model = Idefics2VisionTransformer()
    model.encoder.layers = model.encoder.layers[:-1]
    setattr(model, "patch_size", 14)
    return model


def get_vision_embedding(all_pixel_values, patch_sizes):
    patch_sizes = torch.vstack(patch_sizes)
    max_patches = torch.max(patch_sizes[:, 0] * patch_sizes[:, 1])

    all_pixel_values = torch.nn.utils.rnn.pad_sequence(
        all_pixel_values, batch_first=True, padding_value=0.0)

    B, L, _ = all_pixel_values.shape
    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
    patch_attn_mask = torch.zeros(
        (B, 1, max_patches), dtype=torch.bool)
    vision_embeddings = []
    for i in range(B):
        patch_attn_mask[i, :patch_sizes[i][0] * patch_sizes[i][1]] = True

        vision_embedding = vision_model(
            all_pixel_values, patch_attention_mask=patch_attn_mask).last_hidden_state
        vision_embeddings.append(vision_embedding)
    return vision_embeddings
