"""
ToyNet: An educational framework for building neural networks with NumPy.
"""

__version__ = "0.1.4.2.3"
__author__ = "Ivan Murzin"
__license__ = "MIT"

# Import core components of the package
from ..model import NeuralNetwork
from ..layers import Dense, Flatten
from ..activations import ReLU, Sigmoid, Tanh, LeakyRelu, Softmax
from ..losses import MeanSquaredError, CategoricalCrossEntropy, BinaryCrossEntropy, CrossEntropy, MeanAbsoluteError, QLearningLoss
from ..utils import normalize_data, split_data, one_hot_encode, one_hot_decode, load_mnist
from ..optimizers import GradientDescent, Adam
