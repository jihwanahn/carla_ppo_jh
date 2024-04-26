import os
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTDecoder(nn.Module):
    def __init__(self, latent_dims, nhead, num_decoder_layers, dropout=0.1):
        super().__init__()
        self.model_file = os.path.join('vit/model', 'decoder_model.pth')
        
        self.fc = nn.Linear(latent_dims, 512)
        output_features = 160 * 80 * 3

        decoder_layers = TransformerEncoderLayer(d_model=512, nhead=nhead, dropout=dropout)
        self.transformer_decoder = TransformerEncoder(decoder_layers, num_layers=num_decoder_layers)
        
        self.output_projection = nn.Linear(512, output_features)
        self.final_shape = (3, 80 , 160)

    def forward(self, x):
        x = self.fc(x)
        x = self.transformer_decoder(x.unsqueeze(0))
        x = x.squeeze(0)
        x = self.output_projection(x)
        x = torch.sigmoid(x)
        x = x.view(-1, *self.final_shape)  # Reshape to the desired output dimensions
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
