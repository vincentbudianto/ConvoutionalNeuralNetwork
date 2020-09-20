import numpy as np

class Detector:
    def __init__(self, input = None, activation_function="relu", leaky_slope = 0.5):
        self.input = input
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        self.bias = None

    def activate(self):
        h, w = self.input.shape

        result = np.zeros((self.input.shape))

        for i in range(h):
            for j in range(w):
                result[i][j] = self.forward_activation(self.input[i][j] + self.bias[i][j])

        return result

    def getBias(self):
        return self.bias

    def setBias(self, input):
        self.bias = np.random.randn(input.shape[0], input.shape[1]) * 5

    ###ACTIVATION FUNCTIONS###
    def forward_activation(self, X):
        if self.activation_function == "sigmoid":
            X = np.clip(X, -500, 500)
            return 1.0/(1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0,X)
        elif self.activation_function == "leaky_relu":
            return np.maximum(self.leaky_slope*X,X)

    def get_output(self, inputarray):
        self.sigma = input
        return self.activate()