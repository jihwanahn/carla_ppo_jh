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
from encoder import VariationalEncoder
from decoder import Decoder
from datetime import datetime
import argparse

# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, data_type):
        super(VariationalAutoencoder, self).__init__()
        self.data_type = data_type
        if self.data_type == 'ss':
            self.model_file = os.path.join('autoencoder/model', 'var_autoencoder_ss.pth')
        elif self.data_type == 'rgb':
            self.model_file = os.path.join('autoencoder/model', 'var_autoencoder_rgb.pth')
        # self.model_file = os.path.join('autoencoder/model', 'var_autoencoder.pth')
        self.encoder = VariationalEncoder(latent_dims, data_type)
        self.decoder = Decoder(latent_dims, data_type)

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
        self.encoder.load(self.data_type)
        self.decoder.load(self.data_type)

def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for(x, _) in trainloader:
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = model(x)
        loss = ((x - x_hat)**2).sum() + model.encoder.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss+=loss.item()
    return train_loss / len(trainloader.dataset)


def test(model, testloader):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in testloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = model.encoder(x)
            # Decode data
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() + model.encoder.kl
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='ss', help='Data type: ss or rgb')

    args = parser.parse_args()
    if args.data_type == 'ss':
        data_dir = 'autoencoder/dataset/'
    elif args.data_type == 'rgb':
        data_dir = 'autoencoder/dataset_rgb/'

    writer = SummaryWriter(f"runs/auto-encoder_{args.data_type}/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
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
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE, data_type=args.data_type).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f'Selected device :) :) :) {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model,trainloader, optim)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)
        val_loss = test(model,validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
        print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS,train_loss,val_loss))
    
    model.save()

if __name__ == "__main__":
    main()
