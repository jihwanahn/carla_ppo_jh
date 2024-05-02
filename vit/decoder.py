import os
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTDecoder(nn.Module):
    def __init__(self, latent_dims, nhead, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.model_file = os.path.join('vit/model', 'decoder_model.pth')
        
        width = 160
        height = 80
        self.width = width
        self.height = height

        patch_size = 16
        self.patch_size = patch_size

        self.fc = nn.Linear(latent_dims, 512)
        self.output_projection = nn.ConvTranspose2d(512, 3, kernel_size=patch_size, stride=patch_size)

        decoder_layers = TransformerEncoderLayer(d_model=512, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerEncoder(decoder_layers, num_layers=num_decoder_layers)
        
        self.qs = nn.Parameter(torch.randn((width//patch_size)*(height//patch_size), 512))
        self.final_shape = (3, 80 , 160)

    def forward(self, x):
        #x : batch 95
        batch_size = x.shape[0]
        x = self.fc(x)
        x = x[:, None]
        qs = self.qs[None].repeat(batch_size, 1, 1)
        x = torch.cat([qs, x], dim=1)
        #x : batch 512
        x = self.transformer_decoder(x)
        x = x[:, :(self.width//self.patch_size)*(self.height//self.patch_size)]
        # x : batch 20*40 512
        x = torch.transpose(x, -1, -2)
        # x : batch 512 20*40
        x = x.view(batch_size, -1, self.height//self.patch_size, self.width//self.patch_size)
        x = self.output_projection(x)
        # x : batch 3 80 160
        x = torch.nn.functional.softmax(x, dim=1)
        x = x.view(-1, *self.final_shape)  # Reshape to the desired output dimensions
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))


class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()
        self.model_file = os.path.join('vit/model', 'decoder_model.pth')
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 9 * 4 * 256),
            nn.LeakyReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,4,9))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4,  stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid())
        
    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))