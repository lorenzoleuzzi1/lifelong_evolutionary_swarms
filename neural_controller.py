import numpy as np
import utils

class NeuralController:
    def __init__(self, layer_sizes, weights=None, hidden_activation="relu", output_activation="linear"):
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.activation = hidden_activation
        self.output_activation = output_activation
        assert hidden_activation in ["relu", "sigmoid", "neat_sigmoid", "tanh"],\
            "Activation must be either 'relu' or 'sigmoid' or 'neat_sigmoid' or 'tanh'"
        assert output_activation in ["linear", "sigmoid", "neat_sigmoid", "softmax", "tanh"],\
            "Output activation must be either 'linear' or 'sigmoid' or 'neat_sigmoid' or 'softmax' or 'tanh'" 
        self.total_weights = sum((layer_sizes[i] + 1) * layer_sizes[i+1] for i in range(len(layer_sizes) - 1))
    
    def predict(self, X):
        a = X
        for weight, bias in self.weights[:-1]:  # Forward pass
            z = np.dot(a, weight) + bias
            if self.activation == "relu":
                a = np.maximum(0, z)
            if self.activation == "sigmoid":
                a = 1 / (1 + np.exp(-z)) # Sigmoid activation
            if self.activation == "tanh":
                a = np.tanh(z)
            if self.activation == "neat_sigmoid":
                a = utils.neat_sigmoid(z)
        
        final_weight, final_bias = self.weights[-1]
        output = np.dot(a, final_weight) + final_bias
        
        # Activation for the last layer
        if self.output_activation == "sigmoid":
            output = 1 / (1 + np.exp(-output))
        if self.output_activation == "softmax":
            output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        if self.output_activation == "tanh":
            output = np.tanh(output)
        if self.output_activation == "neat_sigmoid":
            output = utils.neat_sigmoid(output)
        
        return output
    
    def set_weights_from_vector(self, w):
        if w is None:
            raise ValueError(f"None weights provided, expected {self.total_weights} weights.")
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        if len(w) != self.total_weights:
            raise ValueError(f"Expected {self.total_weights} weights, but got {len(w)}")
        
        self.weights = []
        start = 0
        for i in range(len(self.layer_sizes) - 1):
            end = start + self.layer_sizes[i] * self.layer_sizes[i+1]
            weight = w[start:end].reshape(self.layer_sizes[i], self.layer_sizes[i+1])
            start = end
            end = start + self.layer_sizes[i+1]
            bias = w[start:end]
            start = end
            self.weights.append((weight, bias))
    
    def summary(self):
        weights_set = ""
        if self.weights == None:
            weights_set = "not set"
        print("NeuralController with layer sizes: ", self.layer_sizes)
        print(f"Total weights: {self.total_weights}, {weights_set}")