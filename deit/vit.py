import os
import torch
import torch.nn as nn
from vit_pytorch import ViT

# Hyper-parameters
NUM_EPOCHS = 50
BATCH_SIZE = 128#32
LEARNING_RATE = 1e-3#1e-4
LATENT_SPACE = 95  # ResNet18의 fc layer 차원에 맞추어 조정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViTAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(ViTAutoencoder, self).__init__()
        self.model_file = os.path.join('autoencoder/model', 'vit_autoencoder.pth')
        self.encoder = ViT(
            image_size=32,
            patch_size=4,
            num_classes=latent_dims,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
    
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()

def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for(x, _) in trainloader:
        x = x.to(device)
        x_hat = model(x)
        loss = ((x - x_hat)**2).sum() + model.encoder.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(trainloader.dataset)

def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            encoded_data = model.encoder(x)
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() + model.encoder.kl
            val_loss += loss.item()
    return val_loss / len(testloader.dataset)

def main():

    data_dir = 'autoencoder/dataset/'

    writer = SummaryWriter(f"runs/auto-encoder/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    
    # Applying Transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)
    
    m=len(train_data)
    train_data, val_data = random_split(train_data, [int(m-m*0.2), int(m*0.2)])
    

    # Data Loading
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f'Selected device :) :) :) {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model,trainloader, optim)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)
        val_loss = test(model,validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
        print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS,train_loss,val_loss))
    
    model.save()

if __name__ == "__main__":
    main()
