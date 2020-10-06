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

    
    def back_propagation(self, delta_matrix):
        if self.activation_function == "sigmoid":
            ones = np.zeros(delta_matrix.shape) + 1
            return delta_matrix * (ones - delta_matrix)
        elif self.activation_function == "tanh":
            ones = np.zeros(delta_matrix.shape) + 1
            return ones - np.square(np.tanh(delta_matrix))
        elif self.activation_function == "relu":
            return [[1 if col > 0 else 0 for col in row] for row in delta_matrix]
        elif self.activation_function == "leaky_relu":
            return [[self.leaky_slope if col > 0 else 0 for col in row] for row in delta_matrix]

if __name__ == "__main__":
    print("Testing numpy detector layer")

    print("Forward")
    print("Input")
    bias_matrix = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    test_matrix = np.array([[-1, 1, 2, 4], [5, -6, 7, 8], [3, -2, 1, 0], [1, -2, -3, 4]])
    print(test_matrix)
    detector_layer = Detector(test_matrix)
    detector_layer.bias = bias_matrix

    detector_layer.activation_function = "relu"
    print(detector_layer.leaky_slope)
    forward_result = detector_layer.activate()

    print("Result")
    print(forward_result)

    print("Backward")
    print("Input")
    test_delta = np.array([[-1, 1, 2, 4], [5, -6, 7, 8], [3, -2, 1, 0], [1, -2, -3, 4]])
    print(test_delta)
    
    backward_result = detector_layer.back_propagation(test_delta)
    print("Result")
    print(backward_result)


