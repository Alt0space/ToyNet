import os
import struct
import gzip
import numpy as np
import urllib.request


def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def split_data(X, y, train_ratio=0.8):
    size = int(len(X) * train_ratio)
    return X[:size], X[size:], y[:size], y[size:]


def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def one_hot_decode(one_hot):
    return np.argmax(one_hot, axis=1)


def load_mnist(name='digits'):
    base_url = 'http://v21076.hosted-by-vdsina.com:8228/datasets'

    if name == 'digits':
        files = {
            'train_images': 'train-images.idx3-ubyte',
            'train_labels': 'train-labels.idx1-ubyte',
            'test_images': 't10k-images.idx3-ubyte',
            'test_labels': 't10k-labels.idx1-ubyte'
        }

        # Functions to load images and labels
        def load_images(file_path):
            with open(file_path, 'rb') as f:
                _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            return images

        def load_labels(file_path):
            with open(file_path, 'rb') as f:
                _, num_labels = struct.unpack('>II', f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    elif name == 'fashion':
        files = {
            'train_images': 'train-images-idx3-ubyte',
            'train_labels': 'train-labels-idx1-ubyte',
            'test_images': 't10k-images-idx3-ubyte',
            'test_labels': 't10k-labels-idx1-ubyte'
        }

        # Use the same loading functions as for digits
        def load_images(file_path):
            with open(file_path, 'rb') as f:
                _, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
                images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
            return images

        def load_labels(file_path):
            with open(file_path, 'rb') as f:
                _, num_labels = struct.unpack('>II', f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    elif name == 'sign_lang':
        files = {
            'train': 'sign_mnist_train.csv',
            'test': 'sign_mnist_test.csv'
        }

        # Function to load CSV data
        def load_sign_language_csv(file_path):
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
            labels = data[:, 0].astype(int)
            images = data[:, 1:].astype(np.uint8).reshape(-1, 28, 28)
            return images, labels

    else:
        raise ValueError("Dataset not recognized. Please choose 'digits', 'fashion', or 'sign_lang'.")

    # Create a directory to store the data if it doesn't exist
    data_dir = os.path.join(os.getcwd(), 'data', name)
    os.makedirs(data_dir, exist_ok=True)

    # Function to download files
    def download(filename):
        url = f'{base_url}/{name}/{filename}'
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print(f'Downloading {filename}...')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())
        return file_path

    # Download and load datasets
    dataset = {}
    if name in ['digits', 'fashion']:
        for key in files:
            file_path = download(files[key])
            if 'images' in key:
                dataset[key] = load_images(file_path)
            else:
                dataset[key] = load_labels(file_path)
    elif name == 'sign_lang':
        for key in files:
            file_path = download(files[key])
            images, labels = load_sign_language_csv(file_path)
            dataset[f'{key}_images'] = images
            dataset[f'{key}_labels'] = labels

    return dataset

# Example usage:
if __name__ == '__main__':
    # Load digits dataset
    digits_data = load_mnist('sign_lang')
    print('Digits dataset:')
    print('Training images shape:', digits_data['train_images'].shape)
    print('Training labels shape:', digits_data['train_labels'].shape)
    print('Testing images shape:', digits_data['test_images'].shape)
    print('Testing labels shape:', digits_data['test_labels'].shape)

#load_mnist("digits")