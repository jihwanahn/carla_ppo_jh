import os
import torch
import torch.nn as nn

class BiGANDiscriminator(nn.Module):
    def __init__(self, latent_dims, data_dims):
        super(BiGANDiscriminator, self).__init__()
        self.model_file = os.path.join('bigan/model', 'bigan_discriminator.pth')
        data_dims2 = 9216
        self.combined_dims = latent_dims + data_dims  # 잠재 벡터와 데이터의 차원 합

        # 디스크리미네이터 네트워크
        self.discriminator = nn.Sequential(
            nn.Linear(self.combined_dims, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 이진 분류를 위한 Sigmoid 활성화 함수
        )

    def forward(self, x, z):
        # 데이터와 잠재 벡터의 결합
        x = x.view(x.size(0), -1)
        combined_input = torch.cat([x, z], dim=1)
        validity = self.discriminator(combined_input)
        return validity

