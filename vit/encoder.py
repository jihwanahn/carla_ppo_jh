import os
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTEncoder(nn.Module):
    def __init__(self, latent_dims, data_type, nhead, num_encoder_layers, dropout=0.1):
        super(ViTEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.data_type = data_type
        if self.data_type == 'ss':
            self.model_file = os.path.join('vit/model', 'vit_encoder_model_ss.pth')
        elif self.data_type == 'rgb':
            self.model_file = os.path.join('vit/model', 'vit_encoder_model_rgb.pth')
        # self.model_file = os.path.join('vit/model', 'vit_encoder_model.pth')
        
        input_features = 160 * 80 * 3

        width = 160
        height = 80

        patch_size = 16

        self.encode = nn.Conv2d(3, patch_size*patch_size*3, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2)
        self.embedding = nn.Linear(patch_size*patch_size*3, 512)
        
        encoder_layers = TransformerEncoderLayer(d_model=512, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        self.positional_encoding = nn.Parameter(torch.randn((width // patch_size) * (height // patch_size), patch_size*patch_size*3))
        self.q = nn.Parameter(torch.randn(512,))

        self.mu = nn.Linear(512, self.latent_dims)
        self.logvar = nn.Linear(512, self.latent_dims)
        self.output_transform = nn.Linear(512, self.latent_dims)

    def forward(self, x):
        batch_size = x.shape[0]                     # x : batch 3 80 160
        x = self.encode(x)                          # x : batch 16*3 20 40
        x = self.flatten(x)                         # x : batch 16*3 20*40
        x = torch.transpose(x, -1, -2)              # x : batch 20*40 16*3
        x = x + self.positional_encoding[None]
        x = self.embedding(x)                       # x : batch 20*40 512
        q = self.q[None, None].repeat(batch_size, 1, 1)
        x = torch.cat([q, x], dim=1)
        x = self.transformer_encoder(x)             # x : batch 20*40+1 512
        x = x[:, 0]                                 # x : batch 512
        # mu = self.mu(x)
        # logvar = self.logvar(x)

        # combined = torch.cat((mu, logvar), dim=-1)
        # print("Size of mu:", mu.size())             # Size of mu: torch.Size([1, 50])
        # print("Size of logvar:", logvar.size())     # Size of logvar: torch.Size([1, 50])
        # print("Combined size:", combined.size())    # Combined size: torch.Size([1, 100])
        output = self.output_transform(x)
        return output #combined

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        print(f"Loading model from {self.model_file}")
