import sys
import torch
from autoencoder.encoder import VariationalEncoder
from cnn.encoder import CNNEncoder
from vit.encoder import ViTEncoder
from bigan.encoder import BiGANEncoder

#run_name
class EncodeState():
    def __init__(self, latent_dim, run_name):
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_name = run_name

        try:
            if self.run_name == "VAE":
                self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
                self.conv_encoder.load()
                self.conv_encoder.eval()

            elif self.run_name == "CNN":
                self.conv_encoder = CNNEncoder(self.latent_dim).to(self.device)
                self.conv_encoder.load()
                self.conv_encoder.eval()

            elif self.run_name == "VIT":
                self.conv_encoder = ViTEncoder(self.latent_dim, nhead=8, num_encoder_layers=3, dropout=0.1).to(self.device)
                self.conv_encoder.load()
                self.conv_encoder.eval()
            
            elif self.run_name == "BIGAN":
                print("BIGAN not implemented yet.")

            else:
                pass


            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except Exception as e:
            print('Encoder could not be initialized.')
            raise e
    
    def process(self, observation):
        image_obs = torch.tensor(observation[0], dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs = self.conv_encoder(image_obs)
        navigation_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
        observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
        
        return observation