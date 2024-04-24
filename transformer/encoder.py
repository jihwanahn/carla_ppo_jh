import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TransformerEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(TransformerEncoder, self).__init__()
        self.model_file = os.path.join('transformer/model', 'transformer_encoder.pth')