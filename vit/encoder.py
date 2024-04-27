import os
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTEncoder(nn.Module):
    def __init__(self, latent_dims, nhead, num_encoder_layers, dropout=0.1):
        super(ViTEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.model_file = os.path.join('vit/model', 'vit_encoder_model.pth')
        
        input_features = 160 * 80 * 3

        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(input_features, 512)
        
        encoder_layers = TransformerEncoderLayer(d_model=512, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        self.fc_mu = nn.Linear(512, latent_dims)
        self.fc_logvar = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = self.flatten(x)
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(0))
        x = x.squeeze(0)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
