import numpy as np


class MeanAbsoluteError:
    def compute(self, predictions, targets):
        return np.mean(np.abs(predictions - targets))

    def gradient(self, predictions, targets):
        return np.sign(predictions - targets) / targets.size


class MeanSquaredError:
    def compute(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def gradient(self, predictions, targets):
        return 2 * (predictions - targets) / targets.size


class CrossEntropy:
    def compute(self, predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    def gradient(self, predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -(targets / predictions - (1 - targets) / (1 - predictions))


class BinaryCrossEntropy:
    def compute(self, predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

    def gradient(self, predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -(targets / predictions - (1 - targets) / (1 - predictions))


class CategoricalCrossEntropy:
    def compute(self, predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))

    def gradient(self, predictions, targets):
        return predictions - targets


class QLearningLoss:
    def compute(self, q_values, actions, rewards, next_q_values, gamma=0.99):
        targets = rewards + gamma * np.max(next_q_values, axis=1)
        return np.mean((q_values - targets) ** 2)

    def gradient(self, q_values, actions, rewards, next_q_values, gamma=0.99):
        targets = rewards + gamma * np.max(next_q_values, axis=1)
        return 2 * (q_values - targets) / targets.size
