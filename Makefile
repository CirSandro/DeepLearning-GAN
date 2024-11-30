.PHONY: mnist fashionmnist cifar10 all

install:
	pip install -r requirements.txt

run:
	python3 main.py

mnist:
	sed -i "s/DATASET_NAME = .*/DATASET_NAME = 'MNIST'/" config.py
	sed -i "s/IMAGE_SIZE = .*/IMAGE_SIZE = 28/" config.py
	python3 main.py

fashionmnist:
	sed -i "s/DATASET_NAME = .*/DATASET_NAME = 'FashionMNIST'/" config.py
	sed -i "s/IMAGE_SIZE = .*/IMAGE_SIZE = 28/" config.py
	python3 main.py

cifar10:
	sed -i "s/DATASET_NAME = .*/DATASET_NAME = 'CIFAR10'/" config.py
	sed -i "s/IMAGE_SIZE = .*/IMAGE_SIZE = 32/" config.py
	python3 main.py

all:
	make install
	make mnist
	make fashionmnist
	make cifar10
