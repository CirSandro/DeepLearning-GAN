import torch
import torch.nn as nn

class Maxout(nn.Module):
    def __init__(self, input_dim, output_dim, k=2):
        super(Maxout, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.linear = nn.Linear(input_dim, output_dim * k)
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.output_dim, self.k)
        return torch.max(x, dim=2)[0]

class Discriminator(nn.Module):
    def __init__(self, img_size=28, mult=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * mult, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            Maxout(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.img_size = img_size
        self.mult = mult

    def forward(self, x):
        x = x.view(-1, self.img_size * self.img_size * self.mult)
        return self.model(x)
