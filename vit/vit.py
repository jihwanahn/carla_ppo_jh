import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import the newly defined encoder and decoder
from encoder import ViTEncoder
from decoder import ViTDecoder, CNNDecoder

# Hyper-parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dims, nhead, num_layers, dropout=0.1):
        super(ViTAutoencoder, self).__init__()
        self.encoder = ViTEncoder(latent_dims, nhead, num_layers, dropout)
        self.decoder = ViTDecoder(latent_dims, nhead, num_layers, dropout)
        self.decoder2 = CNNDecoder(latent_dims)

    def forward(self, x):

        # x : (batch_size, 3, 160, 80)
        combined = self.encoder(x)
        # 가정: `combined` 텐서는 첫 번째 절반은 `mu`, 두 번째 절반은 `logvar`
        mid_point = combined.size(1) // 2
        mu = combined[:, :mid_point]
        logvar = combined[:, mid_point:]

        # 나머지 모델 계산을 계속 진행
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / BATCH_SIZE
        return BCE, KLD

    def train_model(self, train_loader, valid_loader, optimizer, epochs):
        writer = SummaryWriter(f"runs/vit/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        for epoch in range(epochs):
            self.train()
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                BCE, KLD = self.loss_function(recon_batch, data, mu, logvar)
                loss = BCE # + KLD * 0.0001
                loss.backward()
                optimizer.step()
            writer.add_scalar("Training Loss/epoch", loss.item(), epoch+1)
            print(f'Epoch {epoch+1}, BCE Loss: {BCE.item()}, KLD Loss: {KLD.item()}')            

            self.eval()
            with torch.no_grad():
                val_loss = 0
                for data, _ in valid_loader:
                    data = data.to(device)
                    recon_batch, mu, logvar = self(data)
                    BCE, KLD = self.loss_function(recon_batch, data, mu, logvar)
                    val_loss += (BCE + KLD).item()
                val_loss /= len(valid_loader.dataset)
                writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

    def save(self):
        torch.save(self.state_dict(), 'vit/model/vit_autoencoder.pth')
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.load_state_dict(torch.load('vit/model/vit_autoencoder.pth'))
        self.encoder.load()
        self.decoder.load()

# Define the main function to run the VAE
def main():
    data_dir = 'autoencoder/dataset/'
    data_dir2 = 'vit/dataset/'
    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)
    
    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - 0.2 * m), int(0.2 * m)])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    # Model setup
    input_dim = (3, 160, 80)
    output_dim = (3, 160, 80)
    latent_dims = 95
    nhead = 8
    num_layers = 3
    dropout = 0.1

    model = ViTAutoencoder(input_dim, output_dim, latent_dims, nhead, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train_model(train_loader, valid_loader, optimizer, NUM_EPOCHS)

    model.save()

if __name__ == '__main__':
    main()
