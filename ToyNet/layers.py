import numpy as np


class Layer:
    def get_parameters(self):
        params = {}
        if hasattr(self, 'weights'):
            params['weights'] = self.weights
        if hasattr(self, 'biases'):
            params['biases'] = self.biases
        return params

    def get_gradients(self):
        grads = {}
        if hasattr(self, 'dweights'):
            grads['weights'] = self.dweights
        if hasattr(self, 'dbiases'):
            grads['biases'] = self.dbiases
        return grads

    def set_parameters(self, parameters):
        if 'weights' in parameters:
            self.weights = parameters['weights']
        if 'biases' in parameters:
            self.biases = parameters['biases']


class Dense(Layer):
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.output = None
        self.dweights = None
        self.dbiases = None

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases
        if self.activation:
            self.output = self.activation.forward(self.output)
        return self.output

    def backward(self, grad):
        if self.activation:
            grad = self.activation.backward(grad)
        self.dweights = np.dot(self.input.T, grad)
        self.dbiases = np.sum(grad, axis=0, keepdims=True)
        return np.dot(grad, self.weights.T)

    def update(self, optimizer):
        # Generate unique parameter IDs
        param_id_weights = f'{id(self)}_weights'
        param_id_biases = f'{id(self)}_biases'

        # Update weights
        self.weights = optimizer.update(param_id_weights, self.weights, self.dweights)
        # Update biases
        self.biases = optimizer.update(param_id_biases, self.biases, self.dbiases)

    def __repr__(self):
        return f"Dense Layer: {self.weights.shape[0]} inputs, {self.weights.shape[1]} outputs"


class Conv2D(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, activation=None):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size  # Assuming square kernels
        self.stride = stride
        self.padding = padding
        self.activation = activation

        # Initialize filters (kernels) and biases
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros((output_channels, 1))
        self.input = None
        self.output = None
        self.dweights = None
        self.dbiases = None

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape

        # Calculate output dimensions
        out_height = int((height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        out_width = int((width - self.kernel_size + 2 * self.padding) / self.stride) + 1

        # Initialize output
        self.output = np.zeros((batch_size, self.output_channels, out_height, out_width))

        # Apply padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        else:
            X_padded = X

        # Perform convolution
        for i in range(out_height):
            for j in range(out_width):
                # Define the slice for this step
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]

                # Convolve the slice with the filters
                for k in range(self.output_channels):
                    self.output[:, k, i, j] = np.sum(X_slice * self.weights[k, :, :, :], axis=(1,2,3)) + self.biases[k]

        if self.activation:
            self.output = self.activation.forward(self.output)

        return self.output

    def backward(self, grad):
        batch_size, channels, height, width = self.input.shape

        # Initialize gradients
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        dX = np.zeros_like(self.input)

        # Apply activation backward if needed
        if self.activation:
            grad = self.activation.backward(grad)

        # Apply padding
        if self.padding > 0:
            X_padded = np.pad(self.input, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
            dX_padded = np.pad(dX, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        else:
            X_padded = self.input
            dX_padded = dX

        out_height = grad.shape[2]
        out_width = grad.shape[3]

        # Compute gradients
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                X_slice = X_padded[:, :, h_start:h_end, w_start:w_end]

                for k in range(self.output_channels):
                    # Update gradients for weights and biases
                    self.dweights[k] += np.sum(X_slice * grad[:, k, i, j][:, None, None, None], axis=0)
                    self.dbiases[k] += np.sum(grad[:, k, i, j], axis=0)

                    # Update gradient for input
                    dX_padded[:, :, h_start:h_end, w_start:w_end] += self.weights[k] * grad[:, k, i, j][:, None, None, None]

        # Remove padding from dX if applied
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        return dX

    def update(self, optimizer):
        # Update weights
        self.weights = optimizer.update(self.weights, self.dweights)
        # Update biases
        self.biases = optimizer.update(self.biases, self.dbiases)

    def __repr__(self):
        return f"Conv2D Layer: {self.input_channels} input channels, {self.output_channels} output channels, kernel size {self.kernel_size}"


class MaxPooling2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size  # Assuming square pool
        self.stride = stride
        self.input = None
        self.output = None
        self.mask = None

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))
        self.mask = np.zeros_like(X)

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                X_slice = X[:, :, h_start:h_end, w_start:w_end]
                max_values = np.max(X_slice, axis=(2,3), keepdims=True)
                self.output[:, :, i, j] = max_values.squeeze()

                # Create mask for backward pass
                temp_mask = (X_slice == max_values)
                self.mask[:, :, h_start:h_end, w_start:w_end] += temp_mask

        return self.output

    def backward(self, grad):
        dX = np.zeros_like(self.input)
        out_height = grad.shape[2]
        out_width = grad.shape[3]

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                grad_slice = grad[:, :, i, j][:, :, None, None]
                dX[:, :, h_start:h_end, w_start:w_end] += self.mask[:, :, h_start:h_end, w_start:w_end] * grad_slice

        return dX

    def update(self, optimizer):
        pass  # No parameters to update

    def __repr__(self):
        return f"MaxPooling2D Layer: pool size {self.pool_size}, stride {self.stride}"


class AveragePooling2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size  # Assuming square pool
        self.stride = stride
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        batch_size, channels, height, width = X.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        self.output = np.zeros((batch_size, channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                X_slice = X[:, :, h_start:h_end, w_start:w_end]
                self.output[:, :, i, j] = np.mean(X_slice, axis=(2,3))

        return self.output

    def backward(self, grad):
        dX = np.zeros_like(self.input)
        out_height = grad.shape[2]
        out_width = grad.shape[3]
        pool_area = self.pool_size * self.pool_size

        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                grad_slice = grad[:, :, i, j][:, :, None, None] / pool_area
                dX[:, :, h_start:h_end, w_start:w_end] += grad_slice

        return dX

    def update(self, optimizer):
        pass  # No parameters to update

    def __repr__(self):
        return f"AveragePooling2D Layer: pool size {self.pool_size}, stride {self.stride}"


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.input_shape)

    def update(self, optimizer):
        pass  # No parameters to update

    def __repr__(self):
        return "Flatten Layer"


class Dropout(Layer):
    def __init__(self, rate=0.5):
        self.rate = rate
        self.mask = None
        self.training = True  # Indicates whether the layer is in training or evaluation mode

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) >= self.rate)
            return X * self.mask / (1 - self.rate)
        else:
            return X

    def backward(self, grad):
        if self.training:
            return grad * self.mask / (1 - self.rate)
        else:
            return grad

    def update(self, optimizer):
        pass  # No parameters to update

    def set_training(self, training):
        self.training = training

    def __repr__(self):
        return f"Dropout Layer: rate {self.rate}"


class BatchNorm(Layer):
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))

        self.cache = None

    def forward(self, X, training=True):
        if training:
            batch_mean = np.mean(X, axis=0, keepdims=True)
            batch_var = np.var(X, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            out = self.gamma * self.X_norm + self.beta

            self.cache = (X, self.X_norm, batch_mean, batch_var)
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_norm + self.beta

        return out

    def backward(self, grad):
        X, X_norm, mean, var = self.cache
        N = X.shape[0]

        X_mu = X - mean
        std_inv = 1.0 / np.sqrt(var + self.epsilon)

        dX_norm = grad * self.gamma
        dvar = np.sum(dX_norm * X_mu * -0.5 * std_inv**3, axis=0, keepdims=True)
        dmean = np.sum(dX_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * X_mu, axis=0, keepdims=True)

        dX = dX_norm * std_inv + dvar * 2.0 * X_mu / N + dmean / N
        dgamma = np.sum(grad * X_norm, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dX

    def update(self, optimizer):
        self.gamma -= optimizer.learning_rate * self.dgamma
        self.beta -= optimizer.learning_rate * self.dbeta

    def __repr__(self):
        return f"BatchNorm Layer: num_features {self.num_features}"


