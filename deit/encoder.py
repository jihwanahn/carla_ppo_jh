import os
import torch
import torch.nn as nn
from vit_pytorch import ViT

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ViTEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(ViTEncoder, self).__init__()
        self.model_file = os.path.join('vit/model', 'vit_encoder_model.pth')

        self.encoder = ViT(
            image_size=(160,80),
            patch_size=4,
            num_classes=latent_dims,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        mu, sigma = self.encoder(x)
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
