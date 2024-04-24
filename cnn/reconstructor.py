import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from encoder import CNNEncoder
from decoder import CNNDecoder
from PIL import Image

# Hyper-parameters
BATCH_SIZE = 1
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def unnormalize(img):
    device = img.device
    
    if img.dim() == 4:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    img = img.mul(std).add(mean)
    img = img.clamp(0, 1)
    return img

class CNNEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(CNNEncoder, self).__init__()
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

def main():
    data_dir = 'autoencoder/dataset/'

    test_transforms = transforms.Compose([transforms.Resize((160, 80)),transforms.ToTensor()])

    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    model = CNNEncoder(latent_dims=LATENT_SPACE).to(device)
    model.load()

    count = 1
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            x_hat = model(x)
            x_hat = x_hat.cpu()
            # if x_hat.dim() == 4 and x_hat.size(0) == 1:
            x_hat = x_hat.squeeze(0)
            
            # x_hat = unnormalize(x_hat)
            
            img = transforms.ToPILImage()(x_hat)

            image_filename = str(count) + '.png'
            img.save('cnn/reconstructed/' + image_filename)
            count +=1

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')