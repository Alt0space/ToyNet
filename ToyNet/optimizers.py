import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.params = dict()  # Changed back to params
        self.decay_rate = None
    def update(self, param_name, weights, gradients):
        raise NotImplementedError("The update method must be implemented by subclasses.")


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.01, decay_rate=0.01):
        super().__init__(learning_rate)
        self.initial_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        
    def update(self, param_id, weights, gradients):
        """
        Update weights using gradient descent with learning rate decay.
        :param param_id: Unique identifier for the parameter.
        :param weights: Current weights.
        :param gradients: Gradients computed during backpropagation.
        :return: Updated weights.
        """
        self.iterations += 1
        
        # Calculate decayed learning rate
        current_lr = self.initial_learning_rate / (1 + self.decay_rate * self.iterations)
        self.learning_rate = current_lr
        
        return weights - current_lr * gradients


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        # Initialize the parent class first
        super().__init__(learning_rate)
        
        # Initialize Adam-specific parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.params = {}  # Changed back to params

    def update(self, param_id, weights, gradients):
        if param_id not in self.params:  # Changed state to params
            self.params[param_id] = {  # Changed state to params
                'm': np.zeros_like(weights),
                'v': np.zeros_like(weights),
                't': 0
            }

        params = self.params[param_id]  # Changed state to params
        params['t'] += 1

        # Update biased first moment estimate
        params['m'] = self.beta1 * params['m'] + (1 - self.beta1) * gradients

        # Update biased second raw moment estimate
        params['v'] = self.beta2 * params['v'] + (1 - self.beta2) * (gradients ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = params['m'] / (1 - self.beta1 ** params['t'])

        # Compute bias-corrected second raw moment estimate
        v_hat = params['v'] / (1 - self.beta2 ** params['t'])

        # Update weights
        updated_weights = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_weights
