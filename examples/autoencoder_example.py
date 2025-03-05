from ToyNet.ToyNet import NeuralNetwork, Dense, ReLU, Sigmoid, Flatten, MeanSquaredError, GradientDescent, load_mnist, Adam
import numpy as np
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
data = load_mnist(name='fashion')
train_images, _ = data['train_images'], data['train_labels']
test_images, _ = data['test_images'], data['test_labels']

# Normalize and flatten the images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape images to be flat
train_images_flat = train_images.reshape(train_images.shape[0], -1)  # Flatten to (n_samples, 784)
test_images_flat = test_images.reshape(test_images.shape[0], -1)     # Flatten to (n_samples, 784)

# Create autoencoder model
autoencoder = NeuralNetwork()
autoencoder.add(Flatten())
autoencoder.add(Dense(784, 512, activation=ReLU()))
autoencoder.add(Dense(512, 512, activation=ReLU())) # Increased from 128 to 512
autoencoder.add(Dense(512, 784, activation=Sigmoid())) # Decoder

# Configure model
autoencoder.set_loss(MeanSquaredError())
autoencoder.set_optimizer(Adam(learning_rate=0.0001))
autoencoder.set_regularization(l2_lambda=0.0001)

# Train the model
print("Training autoencoder...")

autoencoder.train(
    X=train_images_flat,
    y=train_images_flat,  # For autoencoders, input = target
    batch_size=256,
    epochs=300,
    X_val=test_images_flat,
    y_val=test_images_flat
)

# Save the model
autoencoder.save('fashion_autoencoderV4.pkl')

# Function to display original and reconstructed images
def plot_reconstructions(original_images, reconstructed_images, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Test the autoencoder on random samples
num_samples = 5
random_indices = np.random.randint(0, len(test_images_flat), num_samples)
test_samples = test_images_flat[random_indices]

# Get reconstructions
reconstructed_samples = autoencoder.predict(test_samples)

# Plot results
plot_reconstructions(test_samples, reconstructed_samples)


