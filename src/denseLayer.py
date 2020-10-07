import numpy as np
from dense import Dense

# sigmoid = lambda x : 1 / (1 + exp(-x))
# relu = lambda x : x * (x > 0)
# d_sigmoid = lambda x : x * (1 - x)
# d_relu = lambda x : 1 * (x > 0)

class DenseLayer:
    def __init__(self, flatlength, nodeCount = 10):
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

        self.outputs = outputArray







