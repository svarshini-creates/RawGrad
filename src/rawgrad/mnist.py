import os
import tempfile
import gzip
import struct
import urllib.request
from pathlib import Path
import numpy as np
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
def download_file(filename, data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / filename
    if file_path.exists():
        print(f"Already downloaded: {filename}")
        return file_path
    url = BASE_URL + filename
    print(f"Downloading {filename}...")
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=data_dir,
            prefix=filename + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_file = Path(temporary.name)
        urllib.request.urlretrieve(url, temp_file)
        # Publishing the download after it completes successfully.
        os.replace(temp_file, file_path)
    except Exception:
        if temp_file is not None and temp_file.exists():
            temp_file.unlink()
        raise
    return file_path
def load_images(file_path):
    with gzip.open(file_path, "rb") as file:
        magic, count, rows, columns = struct.unpack(
            ">IIII",
            file.read(16),
        )
        if magic != 2051:
            raise ValueError("Invalid MNIST image file.")
        images = np.frombuffer(
            file.read(),
            dtype=np.uint8,
        )
    images = images.reshape(count, rows, columns)
    return images
def load_labels(file_path):
    with gzip.open(file_path, "rb") as file:
        magic, count = struct.unpack(
            ">II",
            file.read(8),
        )
        if magic != 2049:
            raise ValueError("Invalid MNIST label file.")
        labels = np.frombuffer(
            file.read(),
            dtype=np.uint8,
        )
    if len(labels) != count:
        raise ValueError("MNIST label count mismatch.")
    return labels
def load_mnist(data_dir="data/mnist"):
    paths = {
        name: download_file(filename, data_dir)
        for name, filename in FILES.items()
    }
    X_train = load_images(paths["train_images"])
    y_train = load_labels(paths["train_labels"])
    X_test = load_images(paths["test_images"])
    y_test = load_labels(paths["test_labels"])
    # Flatten each 28 x 28 image into 784 features.
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)
    # Convert pixel values from 0–255 to 0–1.
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    return X_train, y_train, X_test, y_test