import numpy as np

class Dense:
    def __init__(self, weightarray, activation_function="relu", leaky_slope = 0.5, bias = 0.5):
        self.sigma = None
        self.bias = bias
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        self.weightarray = weightarray

    def calculateSigma(self, inputArray):
        sigmaArray = np.tensordot(self.weightarray, inputArray, (0,0))
        self.sigma = np.sum(sigmaArray) + self.bias

    def activate(self):
        return self.forward_activation(self.sigma)

    def get_weight_with_bias(self):
        return np.append(self.weightarray, self.bias)

    def get_weight(self):
        return self.weightarray

    ###ACTIVATION FUNCTIONS###
    def forward_activation(self, X):
        if self.activation_function == "sigmoid":
            X = np.clip(X, -500, 500)
            return 1.0/(1.0 + np.exp(-X))
        elif self.activation_function == "tanh":
            return np.tanh(X)
        elif self.activation_function == "relu":
            return np.maximum(0,X)
        elif self.activation_function == "softmax":
            return X

    def get_output(self, inputArray):
        self.calculateSigma(inputArray)
        return self.activate()

    def updateWeight(self, newnodeweight):
        self.weightarray = self.weightarray - newnodeweight[:-1]
        self.bias = self.bias - newnodeweight[len(newnodeweight) - 1]