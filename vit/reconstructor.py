import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from vit import ViTAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reconstruct_images(model, data_loader, output_dir ,num_images=2000):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_images:
                break
            data = data.to(device)
            reconstructed = model(data)#, _, _ = model(data)

            # Save original and reconstructed images
            for j in range(data.size(0)):
                # save_image(data[j], os.path.join(output_dir, f'original_{i * data_loader.batch_size + j}.png'))
                save_image(reconstructed[j], os.path.join(output_dir, f'{i * data_loader.batch_size + j}.png'))


def main():
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder('autoencoder/dataset/test', transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize VAE Model
    input_dim = (3, 160, 80)  # Adjust based on your actual input dimensions
    output_dim = (3, 160, 80)
    latent_dims = 95
    nhead = 8
    num_layers = 3
    dropout = 0.1

    model = ViTAutoencoder(input_dim, output_dim, latent_dims, nhead, num_layers, dropout).to(device)
    model_file = os.path.join('vit/model', 'vit_autoencoder.pth')
    # Load model weights if available
    try:
        model.load_state_dict(torch.load('vit/model/vit_autoencoder.pth'))
    except FileNotFoundError:
        print("Model weights not found, ensure the model is trained and weights are saved.")

    output_dir = 'vit/reconstructed/'
    reconstruct_images(model, data_loader, output_dir)

if __name__ == '__main__':
    main()
