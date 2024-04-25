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
from vit.vit import VITAutoencoder
from PIL import Image

# Hyper-parameters
BATCH_SIZE = 1
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    data_dir = 'autoencoder/dataset/'

    test_transforms = transforms.Compose([transforms.ToTensor()])

    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = VITAutoencoder(LATENT_SPACE).to(device)

    model.load()
    count = 1

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_hat = model(x)
            x = x.cpu()
            x_hat = x_hat.cpu()
            x_hat = x_hat.squeeze(0)

            img = transforms.ToPILImage()(x_hat)

            image_filename = str(count) + '.png'
            img.save('vit/reconstructed/' + image_filename)

            count += 1

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        sys.exit()