from functools import partial
from torch.nn.init import trunc_normal_
import numpy as np
import torch
from torch import nn


def pos_embed_val(embed_dim, image_size):
    grid_h_size, grid_w_size = image_size
    grid = np.meshgrid(np.arange(grid_w_size, dtype=np.float32),
                       np.arange(grid_h_size, dtype=np.float32))
    grid = np.stack(grid, axis=0)

    return grid_pos_embed(embed_dim, grid)


def grid_pos_embed(embed_dim, grid):
    emb_h = pos_embed_1d(embed_dim // 2, grid[0])
    emb_w = pos_embed_1d(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=-1)


def pos_embed_1d(embed_dim, pos):
    omega = 1. / 10000 ** (np.arange(embed_dim // 2,
                           dtype=np.float32) / (embed_dim / 2))
    out = np.einsum('hw,d->hwd', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=-1)
    return emb


class Projection(nn.Module):
    def __init__(self, num_queries, embed_dim, num_heads, kv_dim=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), adaptive=False, max_size=(70, 70)):
        super().__init__()
        self.num_queries, self.embed_dim, self.num_heads, self.adaptive, self.max_size = num_queries, embed_dim, num_heads, adaptive, max_size
        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)
        self.kv_proj = nn.Linear(
            kv_dim, embed_dim, bias=False) if kv_dim is not None and kv_dim != embed_dim else nn.Identity()
        self.attn, self.ln_q, self.ln_kv, self.ln_post = nn.MultiheadAttention(
            embed_dim, num_heads), norm_layer(embed_dim), norm_layer(embed_dim), norm_layer(embed_dim)
        self.proj = nn.Parameter(
            (embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))
        self._set_2d_pos_cache(self.max_size)
        self.apply(self._init_weights)

    def _set_2d_pos_cache(self, max_size, device='cpu'):
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed_val(
            self.embed_dim, max_size)).float().to(device), persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h, max_w = torch.max(tgt_sizes[:, 0]), torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]),
                             max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0), nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        bs, device, dtype, patch_len = x.shape[0], x.device, x.dtype, tgt_sizes[:,
                                                                                0] * tgt_sizes[:, 1]
        self._adjust_pos_cache(tgt_sizes, device=device)
        key_padding_mask = torch.zeros(
            (bs, torch.max(patch_len)), dtype=torch.bool, device=device)
        pos_embed = [self.pos_embed[:tgt_sizes[i][0], :tgt_sizes[i][1], :].reshape(
            (patch_len[i], -1)).to(dtype) for i in range(bs)]
        for i in range(bs):
            key_padding_mask[i, patch_len[i]:] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)
        x = self.ln_kv(self.kv_proj(x)).permute(1, 0, 2)
        out = self.attn(self._repeat(self.ln_q(self.query), bs),
                        x + pos_embed, x, key_padding_mask=key_padding_mask)[0]
        return self.ln_post(out.permute(1, 0, 2)) @ self.proj

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
