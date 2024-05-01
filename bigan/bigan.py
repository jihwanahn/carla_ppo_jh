import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from encoder import BiGANEncoder
from generator import BiGANGenerator
from discriminator import BiGANDiscriminator
from datetime import datetime

# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
LATENT_SPACE = 95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiGAN(nn.Module):
    def __init__(self, latent_dims):
        super(BiGAN, self).__init__()
        self.model_file = os.path.join('bigan/model', 'bigan.pth')
        self.encoder = BiGANEncoder(latent_dims)
        self.generator = BiGANGenerator(latent_dims)
        input_dims = 3*160*80
        self.discriminator = BiGANDiscriminator(latent_dims, input_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.generator(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.generator.save()
        self.discriminator.save()
        
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.generator.load()
        self.discriminator.load()

def train(model, trainloader, optim_G, optim_D):
    model.train()
    train_loss = 0.0
    for(x, _) in trainloader:
        x = x.to(device)
        z_real = model.encoder(x)
        x_flatten = x.view(x.size(0), -1)
        combined_input = torch.cat((x_flatten, z_real), dim=-1)
        print(f"Combined input shape: {combined_input.shape}")
        real_validity = model.discriminator(x, z_real)

        z_fake = torch.randn(x.size(0), LATENT_SPACE, device=device)
        fake_imgs = model.generator(z_fake)

        # Debugging the dimensions
        print("Shape of fake_imgs:", fake_imgs.shape)  # Should be similar to x
        print("Shape of z_fake:", z_fake.shape)  # Should match z_real
        
        fake_imgs_flatten = fake_imgs.view(fake_imgs.size(0), -1)
        combined_input_fake = torch.cat((fake_imgs_flatten, z_fake), dim=-1)
        print(f"Combined input fake shape: {combined_input_fake.shape}")

        fake_validity = model.discriminator(fake_imgs.detach(), z_fake)
        
        d_loss = -(torch.log(real_validity) + torch.log(1 - fake_validity)).mean()

        optim_D.zero_grad()
        d_loss.backward()
        optim_D.step()

        g_loss = -torch.log(model.discriminator(fake_imgs, z_fake)).mean()

        optim_G.zero_grad()
        g_loss.backward()
        optim_G.step()

        train_loss += d_loss.item() + g_loss.item()

    return train_loss / len(trainloader.dataset)

def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            z_real = model.encoder(x)
            z_fake = torch.randn(x.size(0), LATENT_SPACE, device=device)
            fake_imgs = model.generator(z_fake)

            real_validity = model.discriminator(x, z_real)
            fake_validity = model.discriminator(fake_imgs, z_fake)
            d_loss = -(torch.log(real_validity) + torch.log(1 - fake_validity)).mean()

            val_loss += d_loss.item()

    return val_loss / len(testloader.dataset)

def main():
    data_dir = 'autoencoder/dataset/'

    writer = SummaryWriter(f"runs/bigan/{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # Applying Transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
    test_dataset = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

    m = len(train_dataset)
    train_dataset, val_dataset = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

    # Data Loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BiGAN(latent_dims=LATENT_SPACE).to(device)
    optimizer_G = torch.optim.Adam(list(model.encoder.parameters()) + list(model.generator.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(model.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))


    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, optimizer_G, optimizer_D)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)

        val_loss = test(model, val_loader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)

        print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS,train_loss,val_loss))

    model.save()

if __name__ == "__main__":
    main()


#     writer.close()
# # 데이터셋 설정
# data_dir = 'autoencoder/dataset/'
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# train_dataset = datasets.ImageFolder(data_dir+'train', transform=transform)
# test_dataset = datasets.ImageFolder(data_dir+'test', transform=transform)

# m = len(train_dataset)

# train_dataset, val_dataset = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# # 모델 및 옵티마이저 초기화
# encoder = BiGANEncoder(latent_dims=LATENT_SPACE).to(device)
# generator = BiGANGenerator(latent_dims=LATENT_SPACE).to(device)
# discriminator = BiGANDiscriminator(latent_dims=LATENT_SPACE, data_dims=3*64*64).to(device)

# optimizer_G = torch.optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr=0.0002, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# # 텐서보드 로그 설정
# writer = SummaryWriter()

# # 학습 루프
# for epoch in range(NUM_EPOCHS):
#     encoder.train()
#     generator.train()
#     discriminator.train()
#     train_loss = 0

#     for imgs, _ in train_loader:
#         real_imgs = imgs.to(device)

#         # 노이즈와 실제 데이터로부터 잠재 벡터 생성
#         z_real = encoder(real_imgs)
#         z_fake = torch.randn(imgs.size(0), 95, device=device)
#         fake_imgs = generator(z_fake)

#         # 디스크리미네이터 훈련
#         real_validity = discriminator(real_imgs, z_real)
#         fake_validity = discriminator(fake_imgs.detach(), z_fake)
#         d_loss = -(torch.log(real_validity) + torch.log(1 - fake_validity)).mean()

#         optimizer_D.zero_grad()
#         d_loss.backward()
#         optimizer_D.step()

#         # 제너레이터 및 인코더 훈련
#         g_loss = -torch.log(discriminator(fake_imgs, z_fake)).mean()

#         optimizer_G.zero_grad()
#         g_loss.backward()
#         optimizer_G.step()

#         train_loss += d_loss.item() + g_loss.item()

#     # 에포크별 훈련 손실 기록
#     avg_train_loss = train_loss / len(train_loader)
#     writer.add_scalar('Loss/Train', avg_train_loss, epoch)

#     # 검증 손실 계산
#     encoder.eval()
#     generator.eval()
#     discriminator.eval()
#     val_loss = 0

#     with torch.no_grad():
#         for imgs, _ in val_loader:
#             real_imgs = imgs.to(device)
#             z_real = encoder(real_imgs)
#             z_fake = torch.randn(imgs.size(0), 95, device=device)
#             fake_imgs = generator(z_fake)

#             real_validity = discriminator(real_imgs, z_real)
#             fake_validity = discriminator(fake_imgs, z_fake)
#             d_loss = -(torch.log(real_validity) + torch.log(1 - fake_validity)).mean()

#             val_loss += d_loss.item()

#     avg_val_loss = val_loss / len(val_loader)
#     writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

#     # 에포크별 출력
#     print(f"[Epoch {epoch}/{NUM_EPOCHS}] Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

# # 텐서보드 리소스 해제
# writer.close()
