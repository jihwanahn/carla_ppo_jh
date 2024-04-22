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
from encoder import ResNetEncoder
from decoder import SimpleDecoder
from datetime import datetime
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 128#32
LEARNING_RATE = 1e-3#1e-4
LATENT_SPACE = 512  # ResNet18의 fc layer 차원에 맞추어 조정


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(ResNetAutoencoder, self).__init__()
        self.model_file = os.path.join('cnn/model', 'resnet_autoencoder.pth')
        self.encoder = ResNetEncoder(latent_dims)
        self.decoder = SimpleDecoder(latent_dims)

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

def train(model, trainloader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    for (x, _) in trainloader:
        x = x.to(device)
        x_hat = model(x)
        # loss = F.mse_loss(x_hat, x, reduction='sum')
        loss = criterion(x_hat, x)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
    return train_loss / len(trainloader.dataset)

def test(model, testloader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            # encoded_data = model.encoder(x)
            x_hat = model(x)
            # loss = F.mse_loss(x_hat, x, reduction='sum')
            loss = criterion(x_hat, x)

            val_loss += loss.item() * x.size(0)

    return val_loss / len(testloader.dataset)

def main():
    data_dir = 'autoencoder/dataset/'
    writer = SummaryWriter(f"runs/ResNet-auto-encoder/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정 추가
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    train_data = datasets.ImageFolder(data_dir + 'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + 'test', transform=test_transforms)

    m = len(train_data)
    train_data, val_data = random_split(train_data, [int(m - m * 0.2), int(m * 0.2)])

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = ResNetAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)

    print(f'Selected device: {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optimizer, criterion)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        val_loss = test(model, validloader, criterion)
        scheduler.step(val_loss)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)
        print('\nEPOCH {}/{} \t LR {:.5f} \t train loss {:.5f} \t val loss {:.5f}'.format(epoch + 1, NUM_EPOCHS, optimizer.param_groups[0]['lr'], train_loss, val_loss))

    # Save the model
    # torch.save(model.state_dict(), 'cnn/model/resnet_autoencoder.pth')
    model.save()

if __name__ == "__main__":
    main()
