import torch
import torch.nn as nn
from einops import rearrange

# DeiT-style patch embedding
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        return x.transpose(1, 2)

class TransformerBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class DeiTTiny(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size=4, embed_dim=192)
        num_patches = (32 // 4) * (32 // 4)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, 192))

        self.blocks = nn.Sequential(*[
            TransformerBlock(dim=192, heads=3) for _ in range(6)
        ])

        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, num_classes)
        self.head_dist = nn.Linear(192, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)

        x = torch.cat((cls, dist, x), dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        cls_out = self.head(x[:, 0])
        dist_out = self.head_dist(x[:, 1])

        return (cls_out + dist_out) / 2
