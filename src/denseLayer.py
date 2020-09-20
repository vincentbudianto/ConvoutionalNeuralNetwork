import numpy as np
from dense import Dense

class DenseLayer:
    def __init__(self, flatlength, nodeCount = 10):
        self.flatArray = None
        self.flatlength = flatlength
        self.denseNodes = []
        self.nodeCount = nodeCount
        self.outputs = None
        self.outputNode = Dense( np.random.randn(self.nodeCount, 1, 1) * 10, activation_function = "relu")

    def initiateLayer(self):
        for _ in range(self.nodeCount):

            currentNodeWeightMatrix = np.random.randn(self.flatlength) * 10
            current_node = Dense(currentNodeWeightMatrix)
            self.denseNodes.append(current_node)

    def executeDenseLayer(self, flatArray):
        outputArray = np.array([])
        self.flatArray = flatArray
        for currentNode in self.denseNodes:
            outputArray = np.append(outputArray, currentNode.get_output(flatArray))


        self.outputs = self.outputNode.get_output(outputArray)

        print(self.outputs)







