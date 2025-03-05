
# ToyNet


ToyNet is a lightweight neural network framework implemented in pure NumPy, designed for educational purposes. It provides a clear and intuitive implementation of deep learning fundamentals, making it an excellent resource for understanding how neural networks work under the hood.

## Features

### Layer Types
- `Dense`: Fully connected layers
- `Conv2D`: Convolutional layers for image processing
- `MaxPooling`: Downsampling operations
- `Flatten`: Reshape layer for connecting Conv2D to Dense layers
- `Dropout`: Regularization through random neuron deactivation

### Activation Functions
- `ReLU`: Rectified Linear Unit
- `Sigmoid`: Logistic function
- `Tanh`: Hyperbolic tangent
- `LeakyReLU`: Variant of ReLU with small negative slope
- `Softmax`: For multi-class classification outputs

### Loss Functions
- `MeanSquaredError`: For regression tasks
- `BinaryCrossEntropy`: For binary classification
- `CategoricalCrossEntropy`: For multi-class classification
- `MeanAbsoluteError`: L1 loss function
- `QLearningLoss`: For reinforcement learning

### Optimizers
- `GradientDescent`: Standard gradient descent optimizer
- `Adam`: Adaptive Moment Estimation optimizer

### Additional Features
- L1/L2 Regularization
- Data preprocessing utilities
- MNIST dataset loader

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ToyNet.git
cd ToyNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Neural Network
```python
from ToyNet import NeuralNetwork, Dense, ReLU, Sigmoid
from ToyNet import MeanSquaredError, Adam

# Create model
model = NeuralNetwork()
model.add(Dense(input_size=784, output_size=128, activation=ReLU()))
model.add(Dense(input_size=128, output_size=10, activation=Sigmoid()))

# Configure model
model.set_loss(MeanSquaredError())
model.set_optimizer(Adam(learning_rate=0.001))

# Train
model.train(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.forward(X_test)
```

### Autoencoder Example
```python
from ToyNet import NeuralNetwork, Dense, ReLU, Sigmoid
from ToyNet import MeanSquaredError, Adam
from ToyNet.utils import load_mnist

# Load and preprocess data
X_train, _ = load_mnist()['train_images'], load_mnist()['train_labels']
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0

# Create autoencoder
model = NeuralNetwork()
model.add(Dense(784, 128, activation=ReLU()))
model.add(Dense(128, 64, activation=ReLU()))
model.add(Dense(64, 128, activation=ReLU()))
model.add(Dense(128, 784, activation=Sigmoid()))

model.set_loss(MeanSquaredError())
model.set_optimizer(Adam(learning_rate=0.001))

# Train
model.train(X_train, X_train, epochs=10, batch_size=32)
```

## Project Structure
```
ToyNet/
├── ToyNet/
│   ├── __init__.py
│   ├── model.py       # Core NeuralNetwork class
│   ├── layers.py      # Layer implementations
│   ├── activations.py # Activation functions
│   ├── losses.py      # Loss functions
│   ├── optimizers.py  # Optimization algorithms
│   └── utils.py       # Helper functions
├── examples/
│   └─── autoencoder_example.py
│   
├── requirements.txt
└── README.md
```

## Dependencies
- NumPy
- tqdm (for progress bars)
- Matplotlib (for examples)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Ivan Murzin

## Acknowledgments
- This project is inspired by modern deep learning frameworks while maintaining simplicity for educational purposes
- Special thanks to the NumPy community for providing the foundation for this implementation
```

This README provides:
1. Clear project description
2. Feature list
3. Installation instructions
4. Usage examples
5. Project structure
6. Dependencies
7. Contributing and license information

You can customize it further by:
- Adding more examples
- Including performance benchmarks
- Adding documentation links
- Including test coverage information
- Adding contact information
- Including a roadmap for future features

