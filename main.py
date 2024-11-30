import torch
import torch.optim as optim
from models.discriminator import Discriminator
from models.generator import Generator
from utils.image_utils import generate_images
from data.dataset import load_data
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LATENT_DIM, IMAGE_SIZE, DATASET_NAME, DEVICE
import matplotlib.pyplot as plt
from torch.optim import Adam
import os
from utils.image_utils import generate_images, plot_losses

device = torch.device(DEVICE)

# load data
train_loader = load_data(DATASET_NAME, BATCH_SIZE, IMAGE_SIZE)

mult = 1
if DATASET_NAME == "CIFAR10":
    mult = 3

# init
discriminator = Discriminator(img_size=IMAGE_SIZE, mult=mult).to(DEVICE)
generator = Generator(latent_dim=LATENT_DIM, img_size=IMAGE_SIZE, mult=mult).to(DEVICE)

criterion = torch.nn.BCELoss()
optimizer_d = Adam(discriminator.parameters(), lr=LEARNING_RATE)
optimizer_g = Adam(generator.parameters(), lr=LEARNING_RATE)

# train
d_losses = []
g_losses = []
k_steps = 5
change_loss_epoch = 5
os.makedirs("results", exist_ok=True)
for epoch in range(NUM_EPOCHS):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1).to(DEVICE)
        fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

        # discriminator (k_steps)
        for _ in range(k_steps):
            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(real_images), real_labels)
            noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_images = generator(noise)
            fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

        # generator
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
        fake_images = generator(noise)

        fake_loss_g = criterion(discriminator(fake_images), real_labels)
        if epoch < change_loss_epoch:
            fake_loss_g = -torch.mean(torch.log(discriminator(fake_images) + 1e-8))
        else:
            fake_loss_g = criterion(discriminator(fake_images), real_labels)

        fake_loss_g.backward()
        optimizer_g.step()

    # save loss
    d_losses.append(d_loss.item())
    g_losses.append(fake_loss_g.item())

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {fake_loss_g.item():.4f}")
    generate_images(generator, epoch, DEVICE, DATASET_NAME, n_images=5, test=mult)

plot_losses(d_losses, g_losses, DATASET_NAME)
