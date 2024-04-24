import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNNDecoder(nn.Module):
    def __init__(self, latent_dims):
        super().__init__()
        self.model_file = os.path.join('cnn/model', 'simple_decoder.pth')
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, 256 * 7 * 7),
            nn.LeakyReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),  # 배치 정규화 추가
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),  # 배치 정규화 추가
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),  # 배치 정규화 추가
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),  # 배치 정규화 추가
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=(0,0)),
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