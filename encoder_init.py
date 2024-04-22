import sys
import torch
from autoencoder.encoder import VariationalEncoder
from cnn.encoder import CNNEncoder
from transformer.encoder import TransformerEncoder
import numpy as np

class EncodeState():
    def __init__(self, run_name='ppo_vae', latent_dim=64):
        self.run_name = run_name
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if self.run_name == "ppo_vae":
                self.encoder = VariationalEncoder(self.latent_dim).to(self.device)
                self.encoder.load()
                # self.conv_encoder.eval()

                # for params in self.conv_encoder.parameters():
                #     params.requires_grad = False
            elif self.run_name == "ppo_cnn":
                pass
            elif self.run_name == "ppo_transformer":
                pass
            
            self.encoder.eval()
            for params in self.encoder.parameters():
                    params.requires_grad = False
        
        except:
            print('Encoder could not be initialized.')

    def process_image(self, image):
        image_obs = torch.tensor(image, dtype=torch.float).to(self.device)
        image_obs = image_obs.unsqueeze(0)
        image_obs = image_obs.permute(0,3,2,1)
        image_obs_result = self.conv_encoder(image_obs)
        
        '''
        for t in range(1000):
            if torch.any(torch.isinf(image_obs_result)):
                image_obs_result = self.conv_encoder(image_obs)
            else:
                break

        if torch.any(torch.isinf(image_obs_result)):
            im = Image.fromarray(observation[0].reshape(observation[0].shape[1], observation[0].shape[0], observation[0].shape[2]))
            im.save('err_img.png')
            raise RuntimeError
        else:
            im = Image.fromarray(observation[0].reshape(observation[0].shape[1], observation[0].shape[0], observation[0].shape[2]))
            im.save('img.png')
        '''

        return image_obs_result
    
    def process(self, image_obs_result, navigation_observation):
        navigation_obs = torch.tensor(navigation_observation, dtype=torch.float).to(self.device)
        result = torch.cat((image_obs_result.view(-1), navigation_obs), -1)
        
        return result
