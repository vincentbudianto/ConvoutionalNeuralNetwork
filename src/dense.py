import numpy as np

class Dense:
    def __init__(self, sigma = None, activation_function="relu", leaky_slope = 0.5):
        self.sigma = sigma
        self.activation_function = activation_function
        self.leaky_slope = leaky_slope

    def activate(self):
        h, w = self.sigma.shape

        result = np.zeros((self.sigma.shape))

        for i in range(0, h):
            for j in range(0, w):
                result[i][j] = self.forward_activation(self.sigma[i][j]) 

        return result              

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

    def get_output(self, inputarray):
        self.sigma = input
        return self.activate()