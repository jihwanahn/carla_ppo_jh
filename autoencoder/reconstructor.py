import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from encoder import VariationalEncoder
from decoder import Decoder
from PIL import Image
import argparse
from tqdm import tqdm

# Hyper-parameters
BATCH_SIZE = 1
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
        self.encoder = VariationalEncoder(latent_dims, self.data_type)
        self.decoder = Decoder(latent_dims, self.data_type)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
    
    def load(self, data_type):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load(data_type=data_type)
        self.decoder.load(data_type=data_type)


def main():
    parser = argparse.ArgumentParser(description='Reconstruct images using the trained VAE model')
    parser.add_argument('--data_type', type=str, default='ss', help='Data type: ss or rgb')

    args = parser.parse_args()
    if args.data_type == 'ss':
        data_dir = 'autoencoder/dataset/'
    elif args.data_type == 'rgb':
        data_dir = 'autoencoder/dataset_rgb/'

    test_transforms = transforms.Compose([transforms.ToTensor()])

    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE, data_type=args.data_type).to(device)
    model.load(data_type=args.data_type)
    count = 1
    os.makedirs(f'autoencoder/reconstructed_{args.data_type}', exist_ok=True)
    with torch.no_grad(): # No need to track the gradients
        for (x, _) in tqdm(testloader):
            # Move tensor to the proper device
            x = x.to(device)
            # Decode data
            x_hat = model(x)
            x_hat = x_hat.cpu()
            x_hat = x_hat.squeeze(0)
            

            # convert the tensor to PIL image using above transform
            img = transforms.ToPILImage()(x_hat)

            image_filename = str(count) +'.png'
            img.save(f'autoencoder/reconstructed_{args.data_type}/'+image_filename)
            count +=1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')
