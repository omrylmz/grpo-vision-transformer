import torch
from torch import nn as nn


# -----------------------------
# 1. Define a Vision Transformer for Binary Classification
# -----------------------------

class VisionTransformerClassifier(nn.Module):
    """
    A simple Vision Transformer (ViT) for binary classification.
    The image is split into patches, projected into an embedding space,
    and passed through a transformer encoder. The class token is then used
    to make a binary prediction.
    """

    def __init__(self, image_size=32, patch_size=4, in_channels=3,
                 embed_dim=64, num_heads=4, num_layers=2, mlp_ratio=4, num_classes=2):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Learnable class token and positional embeddings for patches + class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=int(embed_dim * mlp_ratio),
                                                   dropout=0.1, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)

    def forward(self, x):
        """
        x: (batch, in_channels, image_size, image_size)
        """
        batch_size = x.shape[0]
        # Unfold image into patches: result shape -> (batch, num_patches, patch_dim)
        patches = x.unfold(2, x.shape[2] // (x.shape[2] // 4), x.shape[2] // (x.shape[2] // 4)) \
            .unfold(3, x.shape[3] // (x.shape[3] // 4), x.shape[3] // (x.shape[3] // 4))
        patches = patches.contiguous().view(batch_size, self.num_patches, -1)

        # Project patches to embedding space
        patch_embeddings = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        # Prepend the class token and add positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, patch_embeddings), dim=1)  # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        x = x.transpose(0, 1)  # Transformer expects shape (seq_len, batch, embed_dim)
        x = self.transformer_encoder(x)

        # Use the output corresponding to the class token
        cls_out = x[0]
        logits = self.mlp_head(cls_out)
        return logits
