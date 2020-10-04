import numpy as np
from dense import Dense
from scipy.special import softmax

class OutputLayer:
    def __init__(self, flatlength, nodeCount = 2):
        self.flatArray = None
        self.flatlength = flatlength
        self.denseNodes = []
        self.nodeCount = nodeCount
        self.outputs = None

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

        self.outputs = softmax(outputArray)







