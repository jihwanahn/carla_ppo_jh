import os
import torch
import torch.nn as nn

class BiGANGenerator(nn.Module):
    def __init__(self, latent_dims):
        super(BiGANGenerator, self).__init__()
        self.model_file = os.path.join('bigan/model', 'bigan_generator.pth')  # 모델 파일 경로

        # 레이어 설정
        self.generator_layer1 = nn.Sequential(
            nn.Linear(latent_dims, 256),
            nn.ReLU())

        self.generator_layer2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU())

        self.generator_layer3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU())

        self.generator_layer4 = nn.Sequential(
            nn.Linear(1024, 8*8*256),
            nn.ReLU())

        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU())

        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU())

        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # 64x64
            nn.Tanh())  # 이미지 출력을 위한 Tanh 활성화 함수

    def forward(self, z):
        z = self.generator_layer1(z)
        z = self.generator_layer2(z)
        z = self.generator_layer3(z)
        z = self.generator_layer4(z)
        z = z.view(-1, 256, 8, 8)  # 차원 재조정
        z = self.deconv_layer1(z)
        z = self.deconv_layer2(z)
        z = self.deconv_layer3(z)
        return z

