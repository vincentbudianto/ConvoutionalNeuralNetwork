import numpy as np

class Dense:
    def __init__(self, weightmatrix, activation_function="relu", leaky_slope = 0.5, bias = 0):
        self.sigma = None
        self.bias = bias
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope
        self.weightmatrix = weightmatrix

    def calculateSigma(self, inputArray, inputNode):
        if isinstance(inputArray, list):
            sigmaArray = self.weightmatrix[inputNode] * inputArray
        else:
            sigmaArray = self.weightmatrix[inputNode] * inputArray.flatten()
        self.sigma = np.sum(sigmaArray) + self.bias

    def activate(self):
        return self.forward_activation(self.sigma) 

    ###ACTIVATION FUNCTIONS###
    def forward_activation(self, X):     
        if self.activation_function == "sigmoid":      
            return 1.0/(1.0 + np.exp(-X))    
        elif self.activation_function == "tanh":      
            return np.tanh(X)    
        elif self.activation_function == "relu":      
            return np.maximum(0,X)    
        elif self.activation_function == "leaky_relu":      
            return np.maximum(self.leaky_slope*X,X)

    def get_output(self, inputArray, inputNode):
        self.calculateSigma(inputArray, inputNode)
        return self.activate()