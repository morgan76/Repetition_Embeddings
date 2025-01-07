from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules import FeedForward
import torchaudio


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz




class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/
    """
    def __init__(self, patch_size=4, in_chans=1, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = expand_size(patch_size)

        # make it compatible with stem transformer
        if len(patch_size) > 2:
            patch_size = patch_size[-2:]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x (torch.Tensor): input image/LMS, shape (*, in_chans, H, W)

        Returns:
            torch.Tensor: patch-embedded image, shape (*, H / patch_size, W / patch_size, embed_dim)
        """
        *batch_dims, in_chans, height, width = x.size()
        x = self.proj(x.view(-1, in_chans, height, width))
        if self.flatten:
            x = x.permute(0, 2, 3, 1)  # channels-last
        x = self.norm(x)
        return x.view(*batch_dims, *x.shape[-3:])
    


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = attn_drop

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Implementation of the Multihead Attention compatible with Flash Attention and Nested Tensors.

        Args:
            x (torch.Tensor): input tensor, shape (batch_size, length, embed_dim).
                The length can be variable if a NestedTensor is passed.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv  # each one has shape batch_size, num_heads, seq_length, embed_dim

        x = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.attn_drop if self.training else 0.,
                                           scale=self.scale).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer encoder (M2D) implementation based on the MAE.
    """

    def __init__(self, img_size=224, patch_size=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size, self.patch_size = expand_size(img_size), expand_size(patch_size)
        self.grid_size = [s // p for s, p in zip(self.img_size, self.patch_size)]

        self.pos_embed = nn.Parameter(torch.zeros(1, *self.grid_size, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            FlashAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])

        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=22050,
                                                        n_fft=1024,
                                                        f_min=0,
                                                        f_max=11025,
                                                        n_mels=64, 
                                                        hop_length=256,
                                                        power=2)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.patch_embedding = nn.Conv2d(
            in_channels=1,  # Single-channel mel-spectrogram
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        self.linear = nn.Linear(embed_dim, 128)
        #self.initialize_weights()
        self.positional_encoding = nn.Parameter(
            torch.randn(1, (64 // patch_size) * (64 // patch_size), embed_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.input_bn = nn.BatchNorm2d(1)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        embed_dim = self.pos_embed.size(-1)
        pos_embed = get_sincos_pos_embed(embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, return_layers: bool = False):
        r"""

        Args:
        """

        x = self.spec(x)
        #x = x.permute(0, 2, 1)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)
        x = self.input_bn(x)
        
        patches = self.patch_embedding(x)  # (batch_size, dim, freq_patches, time_patches)
        patches = patches.flatten(2).transpose(1, 2)  # (batch_size, num_patches, dim)
        batch_size, num_patches, _ = patches.size()

        # Add positional encoding
        patches += self.positional_encoding[:, :num_patches]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        cls_output = x[:, 0]
        cls_output = self.linear(cls_output)
        return cls_output
    




def get_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    if len(grid_size) == 1:
        return get_1d_sincos_pos_embed(embed_dim, grid_size[0])

    if len(grid_size) == 2:
        return get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=cls_token)

    if len(grid_size) == 3:
        return get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=cls_token)


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    gS, gH, gW = grid_size
    grid_s = np.arange(gS, dtype=np.float32)
    grid_h = np.arange(gH, dtype=np.float32)
    grid_w = np.arange(gW, dtype=np.float32)
    grid = np.meshgrid(grid_h, grid_s, grid_w)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, gS, gH, gW])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_s = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W, D/2)
    emb = np.concatenate([emb_s, emb_h, emb_w], axis=-1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    gH, gW = grid_size
    grid_h = np.arange(gH, dtype=np.float32)
    grid_w = np.arange(gW, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, gH, gW])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(length))


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = pos[..., np.newaxis] * omega

    # pos = pos.reshape(-1)  # (M,)
    # out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed