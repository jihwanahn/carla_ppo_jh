import os
import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Decoder(nn.Module):
    
    def __init__(self, latent_dims, data_type):
        super().__init__()
        if data_type == 'ss':
            self.model_file = os.path.join('autoencoder/model', 'decoder_model_ss.pth')
        elif data_type == 'rgb':
            self.model_file = os.path.join('autoencoder/model', 'decoder_model_rgb.pth')
        # self.model_file = os.path.join('autoencoder/model', 'decoder_model.pth')
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

    def load(self, data_type):
        if data_type == 'ss':
            self.model_file = os.path.join('autoencoder/model', 'decoder_model_ss.pth')
        elif data_type == 'rgb':
            self.model_file = os.path.join('autoencoder/model', 'decoder_model_rgb.pth')
        print(f"Loading model from {self.model_file}")
        self.load_state_dict(torch.load(self.model_file))