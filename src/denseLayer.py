import numpy as np
from dense import Dense

class DenseLayer:
    def __init__(self, flatlength, nodeCount = 10):
        self.flatArray = None
        self.flatlength = flatlength
        self.denseNodes = []
        self.nodeCount = nodeCount
        self.outputNode = Dense( np.random.randn(self.nodeCount, 1, 1) * 10, activation_function = "sigmoid")

    def initiateLayer(self):
        for _ in range(self.nodeCount):
            
            currentNodeWeightMatrix = np.random.randn(1, self.flatlength) * 10
            current_node = Dense(currentNodeWeightMatrix)
            self.denseNodes.append(current_node)

    def executeDenseLayer(self, flatArray):
        outputArray = []
        self.flatArray = flatArray
        for currentNode in self.denseNodes:
            outputArray.append(currentNode.get_output(flatArray))
        
        print(self.outputNode.get_output(outputArray))
        





        
        