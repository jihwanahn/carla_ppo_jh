import os
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNNEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(CNNEncoder, self).__init__()
        self.model_file = os.path.join('cnn/model', 'resnet_encoder.pth')
        
        original_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # 마지막 fc layer 제외
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(original_model.fc.in_features, latent_dims)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)
    
    def load(self):
        self.load_state_dict(torch.load(self.model_file))