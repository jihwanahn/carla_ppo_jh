import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.model_file = os.path.join('cnn/model', 'simple_decoder.pth')
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 256 * 7 * 7),  # Adjust to match an intermediate size
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),  # Correct dimensions to start unflattening
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Resulting in 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Resulting in 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Resulting in 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Resulting in 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # Final size should be 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))

# def main():
#     dummy_input = torch.randn(1, 512)
#     model = SimpleDecoder(latent_dims=512)
#     output = model(dummy_input)
#     print(output.shape)  # Should print torch.Size([1, 3, 224, 224])

# if __name__ == '__main__':
#     main()