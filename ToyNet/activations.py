import numpy as np


class ReLU:
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad):
        grad[self.input <= 0] = 0
        return grad


class Sigmoid:
    def forward(self, X):
        self.input = X
        return 1 / (1 + np.exp(-X))

    def backward(self, grad):
        sig = self.forward(self.input)
        return grad * sig * (1 - sig)


class Tanh:
    def forward(self, X):
        self.input = X
        return np.tanh(X)

    def backward(self, grad):
        return grad * (1 - np.tanh(self.input) ** 2)


class LeakyRelu:
    def forward(self, X):
        self.input = X
        return np.maximum(0.01 * X, X)

    def backward(self, grad):
        grad[self.input <= 0] *= 0.01
        return grad


class Softmax:
    def forward(self, X):
        self.input = X
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output

    def backward(self, grad):
        # Since we combined the gradient in the loss function, we can return grad directly
        return grad