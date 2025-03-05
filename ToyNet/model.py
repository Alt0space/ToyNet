import numpy as np
import pickle
from tqdm import trange


class NeuralNetwork:
    def __init__(self):
        """
        Initializes the neural network.
        """
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.l1_lambda = 0  # L1 regularization strength
        self.l2_lambda = 0  # L2 regularization strength

    def add(self, layer):
        """
        Adds a layer to the network.
        :param layer: The layer to add.
        """
        self.layers.append(layer)

    def set_loss(self, loss):
        """
        Sets the loss function.
        :param loss: The loss function object.
        """
        self.loss = loss

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer.
        :param optimizer: The optimizer object.
        """
        self.optimizer = optimizer

    def set_regularization(self, l1_lambda=0, l2_lambda=0):
        """
        Sets the regularization parameters.
        :param l1_lambda: L1 regularization strength
        :param l2_lambda: L2 regularization strength
        """
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, X, training=True):
        """
        Performs a forward pass through the network.
        :param X: Input data.
        :param training: Boolean indicating training or evaluation mode.
        :return: Output of the network.
        """
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training
            X = layer.forward(X)
        return X

    def backward(self, grad):
        """
        Performs a backward pass through the network.
        :param grad: Gradient from the loss function.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _compute_regularization_loss(self):
        """
        Computes the regularization loss.
        :return: regularization loss value
        """
        reg_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                params = layer.get_parameters()
                if 'weights' in params:
                    weights = params['weights']
                    if self.l1_lambda > 0:
                        reg_loss += self.l1_lambda * np.sum(np.abs(weights))
                    if self.l2_lambda > 0:
                        reg_loss += self.l2_lambda * np.sum(np.square(weights)) / 2
        return reg_loss

    def train(self, X, y, epochs=100, batch_size=32, X_val=None, y_val=None, patience=5):
        """
        Trains the network.
        :param X: Training data.
        :param y: Training labels.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        :param X_val: Validation data.
        :param y_val: Validation labels.
        :param patience: Number of epochs to wait before early stopping.
        """
        if self.loss is None or self.optimizer is None:
            raise ValueError("Loss function and optimizer must be set before training.")

        num_samples = X.shape[0]

        # Add early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in trange(epochs, desc='Training'):
            epoch_losses = []  # Track losses for averaging

            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                predictions = self.forward(X_batch, training=True)
                loss_value = self.loss.compute(predictions, y_batch)
                
                # Add regularization loss
                reg_loss = self._compute_regularization_loss()
                total_loss = loss_value + reg_loss
                
                # Get the gradient from loss function
                grad = self.loss.gradient(predictions, y_batch)
                
                # Backward pass
                self.backward(grad)
                
                # Update weights with regularization
                for layer in self.layers:
                    if hasattr(layer, 'get_parameters'):
                        parameters = layer.get_parameters()
                        gradients = layer.get_gradients()
                        updated_parameters = {}
                        for param_name in parameters:
                            param = parameters[param_name]
                            grad = gradients[param_name]
                            
                            # Add regularization gradients for weights
                            if param_name == 'weights':
                                if self.l1_lambda > 0:
                                    grad += self.l1_lambda * np.sign(param)
                                if self.l2_lambda > 0:
                                    grad += self.l2_lambda * param
                            
                            updated_parameters[param_name] = self.optimizer.update(
                                f'{id(layer)}_{param_name}',
                                param,
                                grad
                            )
                        layer.set_parameters(updated_parameters)

                # Store total loss including regularization
                epoch_losses.append(total_loss)

            # Calculate average epoch loss
            avg_epoch_loss = np.mean(epoch_losses)

            """            # Validation with early stopping
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val, training=False)
                val_loss = self.loss.compute(val_predictions, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break """
   
            print(f"\nEpoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}")
            

    def predict(self, X):
        """
        Makes predictions using the trained network.
        :param X: Input data.
        :return: Predicted output.
        """
        return self.forward(X, training=False)

    def evaluate(self, X, y):
        """
        Evaluates the network on test data.
        :param X: Test data.
        :param y: Test labels.
        :return: Loss and accuracy.
        """
        predictions = self.forward(X, training=False)
        loss_value = self.loss.compute(predictions, y)
        accuracy = self._compute_accuracy(predictions, y)
        print(f"Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}")
        return loss_value, accuracy

    def _compute_accuracy(self, predictions, y):
        """
        Computes the accuracy of predictions.
        :param predictions: Predictions from the network.
        :param y: True labels.
        :return: Accuracy metric.
        """
        if predictions.shape[1] == 1:
            # Binary classification
            predicted_classes = (predictions > 0.5).astype(int)
            true_classes = y
        else:
            # Multi-class classification
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(predicted_classes.flatten() == true_classes.flatten())
        return accuracy

    def save(self, filename):
        """
        Saves the network parameters to a file.
        :param filename: Name of the file.
        """
        parameters = []
        for layer in self.layers:
            if hasattr(layer, 'get_parameters'):
                layer_params = layer.get_parameters()
                # Convert numpy arrays to lists for serialization
                serialized_params = {k: v.tolist() for k, v in layer_params.items()}
                parameters.append({
                    'class_name': layer.__class__.__name__,
                    'params': serialized_params
                })
            else:
                parameters.append(None)
        with open(filename, 'wb') as f:
            pickle.dump(parameters, f)

    def load(self, filename):
        """
        Loads network parameters from a file.
        :param filename: Name of the file.
        """
        with open(filename, 'rb') as f:
            parameters = pickle.load(f)
        for layer, layer_data in zip(self.layers, parameters):
            if layer_data is not None:
                # Convert lists back to numpy arrays
                params = {k: np.array(v) for k, v in layer_data['params'].items()}
                layer.set_parameters(params)

    def __repr__(self):
        """
        Returns a string representation of the network.
        :return: String representation.
        """
        return "\n".join([str(layer) for layer in self.layers])