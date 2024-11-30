# Deep Learning GAN Project

This project implements a Generative Adversarial Network (GAN) using PyTorch, structured as a modular and extensible Python library. The implementation includes options to train and evaluate the GAN on three datasets: **MNIST**, **FashionMNIST**, and **CIFAR10**.

---

## Project Structure

```
DeepLearning-GAN/
├── config.py            # Centralized configuration file
├── data/
│   ├── dataset.py       # Dataset loading logic
├── docs/
│   ├── diapo.pdf        # Project slides
│   ├── DL.pdf           # Project report
├── main.py              # Main script for training
├── models/
│   ├── discriminator.py # Discriminator implementation
│   ├── generator.py     # Generator implementation
├── results/             # Generated images and loss plots
├── utils/
│   ├── image_utils.py   # Image generation and loss visualization
├── requirements.txt     # Python dependencies
├── Makefile             # Commands for dataset selection and training
└── README.md            # Project documentation
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:CirSandro/DeepLearning-GAN.git
   cd DeepLearning-GAN
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   make install
   ```

---

## Usage

### 1. Run Training
You can select the dataset and initiate training using the `Makefile`:

- **MNIST**:
  ```bash
  make mnist
  ```

- **FashionMNIST**:
  ```bash
  make fashionmnist
  ```

- **CIFAR10**:
  ```bash
  make cifar10
  ```

- **Run all datasets sequentially**:
  ```bash
  make all
  ```

### 2. Manual Training
You can also manually modify `config.py` to set the `DATASET_NAME` and `IMAGE_SIZE` before running `main.py`:
```bash
python3 main.py
```

---

## Results

Training generates:
1. **Generated Images**: Saved in `results/<dataset_name>/generated_images_epoch_<epoch>.png`.
2. **Loss Curves**: Saved in `results/<dataset_name>/loss_curves.png`.

---

## Configuration

You can customize the model and training parameters in `config.py`:
```python
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
LATENT_DIM = 100
IMAGE_SIZE = 28
DATASET_NAME = 'MNIST'  # Options: 'MNIST', 'FashionMNIST', 'CIFAR10'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## References

1. **GAN Original Paper**: Goodfellow, Ian, et al. *"Generative Adversarial Nets."* NeurIPS 2014.  
2. **Datasets**:
   - MNIST: http://yann.lecun.com/exdb/mnist/
   - FashionMNIST: https://github.com/zalandoresearch/fashion-mnist
   - CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html

---

## Author

Cardi Julien
Ferroni Sandro
Moyo Kamdem Auren
