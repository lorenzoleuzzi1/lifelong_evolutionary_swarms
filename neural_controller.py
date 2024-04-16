import numpy as np

class NeuralController:
    def __init__(self, layer_sizes, weights=None, activation="relu"):
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.activation = activation
        assert activation in ["relu", "sigmoid"], "Activation must be either 'relu' or 'sigmoid'"
        self.total_weights = sum((layer_sizes[i] + 1) * layer_sizes[i+1] for i in range(len(layer_sizes) - 1))
    
    def predict(self, X):
        a = X
        for weight, bias in self.weights[:-1]:  # Apply ReLU to all layers except the last
            z = np.dot(a, weight) + bias
            if self.activation == "relu":
                a = np.maximum(0, z)
            else:
                a = 1 / (1 + np.exp(-z)) # Sigmoid activation
        
        # Linear activation for the last layer
        final_weight, final_bias = self.weights[-1]
        output = np.dot(a, final_weight) + final_bias
        return output
    
    def set_weights_from_vector(self, w):
        if w is None:
            self.weights = None
            return
        if len(w) != self.total_weights:
            raise ValueError(f"Expected {self.total_weights} weights, but got {len(w)}")
        if not isinstance(w, np.ndarray):
            w = np.array(w)
        
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