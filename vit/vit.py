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
from encoder import ViTEncoder
from decoder import ViTDecoder
from datetime import datetime


# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VITAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VITAutoencoder, self).__init__()
        self.model_file = os.path.join('vit/model', 'vit_autoencoder.pth')
        self.encoder = ViTEncoder(latent_dims, image_size=(160, 80), patch_size=8)
        self.decoder = ViTDecoder(latent_dims, image_size=(160, 80), patch_size=8)

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

def train(model, trainloader, optim, criterion):
    model.train()
    train_loss = 0.0
    for(x, _) in trainloader:
        # Move tensor to the proper device
        x = x.to(device)
        optim.zero_grad()
        
        x_hat = model(x)
        loss = criterion(x, x_hat)
        
        loss.backward()
        optim.step()

        train_loss+=loss.item() * x.size(0)
    return train_loss / len(trainloader.dataset)


def test(model, testloader, criterion):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in testloader:
            # Move tensor to the proper device
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x, x_hat)
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():

    data_dir = 'autoencoder/dataset/'

    writer = SummaryWriter(f"runs/vit/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    # Applying Transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)
    
    m=len(train_data)
    train_data, val_data = random_split(train_data, [int(m-m*0.2), int(m*0.2)])
    

    # Data Loading
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
        
    model = VITAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optim = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    print(f'Selected device :) :) :) {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model,trainloader, optim, criterion)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)
        val_loss = test(model,validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
        print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS,train_loss,val_loss))
    
    model.save()

if __name__ == "__main__":
    main()
