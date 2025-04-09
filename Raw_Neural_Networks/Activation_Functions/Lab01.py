## 1. Import required Libraries
import numpy as np
import matplotlib.pyplot as plt

## 2. Sigmoid function and its derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define Sigmoid derivative, "the sigmoid gradient"
def sigmoid_derivative(z):
    return  sigmoid(z) * (1 - sigmoid(z))

## 3. Define the ReLU function
def relu(z):
    return np.maximum(0, z)

# Define the ReLU derivative (ReLU gradient)
def relu_derivative(z):
    return np.where(z > 0, 1, 0)


## 4. Define the number of input values

# Generate a range of input values
z = np.linspace(-10, 10, 400)
sigmoid_grad = sigmoid_derivative(z)
relu_grad = relu_derivative(z)
# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()