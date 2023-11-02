import numpy as np
import matplotlib.pyplot as plt
import math

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.output = 0

    def feedforward(self, inputs):
        self.output = sigmoid(np.dot(inputs, self.weights) + self.bias)
        return self.output

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.hidden_neurons = [Neuron(num_inputs) for _ in range(num_hidden)]
        self.output_neuron = Neuron(num_hidden)
        self.error_history = []

    def feedforward(self, inputs):
        hidden_outputs = np.array([neuron.feedforward(inputs) for neuron in self.hidden_neurons])
        return self.output_neuron.feedforward(hidden_outputs)

    def backpropagation(self, inputs, target):
        # Feedforward for output
        hidden_outputs = np.array([neuron.feedforward(inputs) for neuron in self.hidden_neurons])
        output = self.output_neuron.feedforward(hidden_outputs)

        # Calculate the error
        error = target - output
        self.error_history.append(error)

        # Calculate deltas
        output_delta = error * sigmoid_derivative(output)

        hidden_deltas = []
        for hidden_neuron, hidden_output in zip(self.hidden_neurons, hidden_outputs):
            hidden_deltas.append(output_delta * self.output_neuron.weights * sigmoid_derivative(hidden_output))

        # Update output neuron weights
        for i, hidden_output in enumerate(hidden_outputs):
            self.output_neuron.weights[i] += output_delta * hidden_output
        self.output_neuron.bias += output_delta

        # Update hidden neuron weights
        for i, hidden_neuron in enumerate(self.hidden_neurons):
            for j in range(len(hidden_neuron.weights)):
                hidden_neuron.weights[j] += hidden_deltas[i][j] * inputs[j]
            hidden_neuron.bias += hidden_deltas[i]

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                self.feedforward(x)
                self.backpropagation(x, y)

# Training the Neural Network
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[1], [1], [1], [0]])  # Targets for NAND
epochs = 10000

# Create the neural network
# Assuming we want 2 hidden neurons for this simple task
nn = NeuralNetwork(num_inputs=2, num_hidden=2, num_outputs=1)

# Train the neural network
nn.train(inputs, targets, epochs)

# Testing the Neural Network
for input in inputs:
    print(f"Input: {input} - Output: {nn.feedforward(input)}")

# Plotting the error over epochs
plt.plot(range(len(nn.error_history)), nn.error_history)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error Trend over Time')
plt.show()
