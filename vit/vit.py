import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Import the newly defined encoder and decoder
from encoder import ViTEncoder
from decoder import ViTDecoder

# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dims, nhead, num_layers, dropout=0.1):
        super(ViTAutoencoder, self).__init__()
        self.encoder = ViTEncoder(latent_dims, nhead, num_layers, dropout)
        self.decoder = ViTDecoder(latent_dims, nhead, num_layers, dropout)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    def train_model(self, train_loader, valid_loader, optimizer, epochs):
        writer = SummaryWriter(f"runs/vit/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        for epoch in range(epochs):
            self.train()
            for data, _ in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

            self.eval()
            with torch.no_grad():
                val_loss = 0
                for data, _ in valid_loader:
                    data = data.to(device)
                    recon_batch, mu, logvar = self(data)
                    val_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                val_loss /= len(valid_loader.dataset)
                writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

# Define the main function to run the VAE
def main():
    data_dir = 'autoencoder/dataset/'
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
    latent_dims = 50
    nhead = 8
    num_layers = 3
    dropout = 0.1

    model = ViTAutoencoder(input_dim, output_dim, latent_dims, nhead, num_layers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train_model(train_loader, valid_loader, optimizer, 50)

if __name__ == '__main__':
    main()
