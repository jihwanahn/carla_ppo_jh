import os
import torch
import torch.nn as nn

class ViTDecoder(nn.Module):
    def __init__(self, latent_dims, image_size=(160, 80), patch_size=8):
        super().__init__()
        self.model_file = os.path.join('vit/model', 'vit_decoder_model.pth')
        
        # Assuming the latent space dimension is 512 as specified in the encoder
        latent_space_dimension = 512
        
        # Define the layers of the decoder
        self.decoder = nn.Sequential(
            # First linear layer to upsample from the encoded latent dimension
            nn.Linear(latent_space_dimension, 512 * (image_size[0] // patch_size) * (image_size[1] // patch_size)),
            nn.ReLU(),
            
            # Reshape into a set of feature maps
            nn.Unflatten(dim=1, unflattened_size=(512, image_size[0] // patch_size, image_size[1] // patch_size)),
            
            # Series of transposed convolutions to upscale the feature maps to the original image size
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Final layer to produce the output image with 3 channels (RGB)
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Sigmoid to ensure pixel values are between 0 and 1
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
