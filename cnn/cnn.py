import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from encoder import CNNEncoder
from decoder import CNNDecoder
from datetime import datetime
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from pytorch_msssim import ssim, MS_SSIM
from tqdm import tqdm

# Hyper-parameters
NUM_EPOCHS = 50#1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4#1e-4
LATENT_SPACE = 95


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(CNNAutoencoder, self).__init__()
        self.model_file = os.path.join('cnn/model', 'resnet_autoencoder.pth')
        self.encoder = CNNEncoder(latent_dims)
        self.decoder = CNNDecoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
        
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()

def ssim_loss(x, x_hat):
    return 1 - ssim(x, x_hat, data_range=1.0, size_average=True)

def train(model, trainloader, optimizer):
    model.train()
    train_loss = 0.0
    for (x, _) in tqdm(trainloader, desc='Training', unit='batch'):
        x = x.to(device)
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x, reduction='mean')
        # loss = ssim_loss(x, x_hat)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
    return train_loss / len(trainloader.dataset)

def test(model, testloader):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, _ in tqdm(testloader, desc='Validation', unit='batch'):
            x = x.to(device)
            x_hat = model(x)
            loss = F.mse_loss(x_hat, x, reduction='mean')
            # loss = ssim_loss(x, x_hat)
            val_loss += loss.item() * x.size(0)
    return val_loss / len(testloader.dataset)

def main():
    data_dir = 'autoencoder/dataset/'
    data_dir2 = 'vit/dataset/'
    writer = SummaryWriter(f"runs/cnn/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    train_transforms = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    
    test_transforms = transforms.Compose([
        transforms.ToTensor()
        ])
    
    train_data = datasets.ImageFolder(data_dir + 'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + 'test', transform=test_transforms)

    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = CNNAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    for epoch in tqdm(range(NUM_EPOCHS), desc='Epoch'):
        train_loss = train(model, trainloader, optimizer)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        val_loss = test(model, validloader)
        scheduler.step(val_loss)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)
        tqdm.write(f'EPOCH {epoch + 1}/{NUM_EPOCHS} \t LR: {optimizer.param_groups[0]["lr"]} \t Train Loss: {train_loss:.5f} \t Val Loss: {val_loss:.5f}')
    print(f'EPOCH {epoch + 1}/{NUM_EPOCHS} \t LR: {optimizer.param_groups[0]["lr"]} \t Train Loss: {train_loss:.5f} \t Val Loss: {val_loss:.5f}')

    model.save()

if __name__ == "__main__":
    main()
