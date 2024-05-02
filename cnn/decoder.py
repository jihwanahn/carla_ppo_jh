import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNNDecoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.model_file = os.path.join('cnn/model', 'simple_decoder.pth')
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 512 * 5 * 3),  # Adjusted to suit the required output dimensions
            nn.LeakyReLU(),
            nn.Unflatten(1, (512, 5, 3)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1), 
            nn.Tanh(),  # Changing from Sigmoid to Tanh for normalized image output
            nn.AdaptiveAvgPool2d((80, 160))  # Ensures the output is exactly 160x80
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