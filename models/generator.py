import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28, mult=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size * img_size * mult),
            nn.Tanh()
        )
        self.img_size = img_size
        self.mult = mult

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, self.mult, self.img_size, self.img_size)
