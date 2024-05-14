import os
import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CNNEncoder(nn.Module):
    def __init__(self, latent_dims, data_type):
        super(CNNEncoder, self).__init__()
        if data_type == 'ss':
            self.model_file = os.path.join('cnn/model', 'cnn_encoder_ss.pth')
        elif data_type == 'rgb':
            self.model_file = os.path.join('cnn/model', 'cnn_encoder_rgb.pth')
        
        resnet18_pretrained = models.resnet18(pretrained=True)
        # self.features = resnet18_pretrained.fc.in_features
        self.features = nn.Sequential(*list(resnet18_pretrained.children())[:-1])  # 마지막 fc layer 제외
        self.dropout = nn.Dropout(0.5)
        # self.dropout = resnet18_pretrained.
        self.fc = nn.Linear(resnet18_pretrained.fc.in_features, latent_dims)
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
    
    def load(self, data_type):
        if data_type == 'ss':
            self.model_file = os.path.join('cnn/model', 'cnn_encoder_ss.pth')
        elif data_type == 'rgb':
            self.model_file = os.path.join('cnn/model', 'cnn_encoder_rgb.pth')
        self.load_state_dict(torch.load(self.model_file))