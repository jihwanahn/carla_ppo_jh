import os
import torch
import torch.nn as nn
from vit_pytorch import ViT
from vit_pytorch.efficient import ViT as EfficientViT

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ViTEncoder(nn.Module):
    def __init__(self, image_size, patch_size, latent_dims, dim, depth, heads, mlp_dim, channels=3):
        super(ViTEncoder, self).__init__()
        self.model_file = os.path.join('vit/model', 'vit_encoder_model.pth')

        self.encoder = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=latent_dims,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
