import torch
import torch.nn as nn
from einops import rearrange

# --- Patch Embedding ---
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

# --- Simple ViT Block ---
class ViTBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# --- Full ViT-Tiny for CIFAR ---
class ViTTinyPatch4(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size=4, embed_dim=192)
        num_patches = (32 // 4) * (32 // 4)  # = 64 patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 192))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 192))

        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim=192, num_heads=3) for _ in range(6)]
        )

        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])
