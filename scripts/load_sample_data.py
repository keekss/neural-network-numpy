import gzip
import os
from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

import logging
from . import logging_config as _

# Main function is prepare_and_load_fashion_mnist_data

# Helper functions -----------------------------------------------------------

# TODO fix path
def download_and_save_files(url: str, file_names: list, path: str) -> None:
    """Download and save files from a specified URL to a local directory."""
    os.makedirs(path, exist_ok=True)  # Create path if it doesn't exist.

    # Download any missing files
    for file in file_names:
        if not os.path.isfile(os.path.join(path, file)):
            urlretrieve(url + file, os.path.join(path, file))
            logging.info(f"Downloaded {file} to {path}")
        else: logging.info(f"File {file} already exists in {path}")

def load_image_data_from_file(path: str) -> np.ndarray:
    """Return images loaded locally."""
    with gzip.open(path) as f:
        # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
        pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 784).astype('float32') / 255

def load_label_data_from_file(path: str) -> np.ndarray:
    """Return labels loaded locally."""
    with gzip.open(path) as f:
        # First 8 bytes are magic_number, n_labels
        integer_labels = np.frombuffer(f.read(), 'B', offset=8)

    return encode_labels_as_onehot(integer_labels)

def encode_labels_as_onehot(integer_labels: np.ndarray) -> np.ndarray:
    """Return matrix whose rows are onehot encodings of integers."""
    n_rows = len(integer_labels)
    n_cols = integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot

def plot_fashion_mnist_images(images: np.ndarray, labels: np.ndarray) -> None:
    """Plot the first 10 images and their labels from the training set."""
    plt.figure(1, figsize=(14,6))
    for i in range(10):
        plt.subplot(1,10, i+1)
        plt.imshow(images[i,:].reshape(28,28), cmap='bone')
        plt.title(f'Label: {labels[i].argmax()}')
        plt.xticks([])
        plt.yticks([])

# Main function --------------------------------------------------------------

def prepare_and_load_fashion_mnist_data(path: str = './data/', plot: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    Download compressed Fashion MNIST data to local directory, and 
    unpack data into numpy arrays. 
    
    Return (train_images, train_labels, test_images, test_labels).

    Args:
        plot (bool): Whether to plot the first 10 images from the training set.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values.
            Columns of labels are a onehot encoding of the correct class.
    """
    url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    file_names = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    # Download files
    download_and_save_files(url, file_names.values(), path)

    # Load data from files
    data = {}
    for alias, file_name in file_names.items():
        file_path = os.path.join(path, file_name)
        if 'images' in alias:
            data[alias] = load_image_data_from_file(file_path)
        else:
            data[alias] = load_label_data_from_file(file_path)

    if plot:
        plot_fashion_mnist_images(data['train_images'], data['train_labels'])
    
    return data