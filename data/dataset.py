import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def load_data(dataset_name, batch_size, image_size, data_dir="./data"):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "FashionMNIST":
        dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR10":
        transform_cifar = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB
        ])
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_cifar)
    else:
        raise ValueError(f"Error dataset not exists : '{dataset_name}'")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
