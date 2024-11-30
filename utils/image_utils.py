import torch
import matplotlib.pyplot as plt

def generate_images(generator, epoch, device, dataset_name, latent_dim=100, n_images=10, test=1):
    generator.eval()
    noise = torch.randn(n_images, latent_dim).to(device)
    generated_images = generator(noise).detach().cpu()

    generated_images = (generated_images + 1) / 2

    fig, axes = plt.subplots(1, n_images, figsize=(15, 3))
    for i in range(n_images):
        if test == 1:
            axes[i].imshow(generated_images[i].squeeze(), cmap="gray")
        else:
            axes[i].imshow(generated_images[i].permute(1, 2, 0))
        axes[i].axis("off")
    plt.savefig(f"results/{dataset_name}/generated_images_epoch_{epoch + 1}.png")
    plt.close()

def plot_losses(d_losses, g_losses, dataset_name):
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig(f"results/{dataset_name}/loss_curves.png")
    plt.close()
