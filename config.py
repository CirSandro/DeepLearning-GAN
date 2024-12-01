import torch

BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
LATENT_DIM = 100
IMAGE_SIZE = 28
DATASET_NAME = 'MNIST'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
