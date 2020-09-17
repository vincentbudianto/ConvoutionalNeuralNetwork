import numpy as np
from dense import Dense

class DenseLayer:
    def __init__(self, previousChannel = 3, previousFeatureMapSize = 50, nodeCount = 10):
        self.featureMap = None
        print("NODE", nodeCount)
        self.previousChannel = previousChannel
        self.previousFeatureMapSize = previousFeatureMapSize
        self.denseNodes = []
        self.nodeCount = nodeCount
        self.outputNode = Dense( np.random.randn(self.nodeCount, 1, 1) * 10, activation_function = "relu")

    def initiateLayer(self):
        
        weightMatrix = np.random.randn(self.previousChannel, self.nodeCount, self.previousFeatureMapSize * self.previousFeatureMapSize) * 10
        
        for i in range(self.nodeCount):
            currentNodeWeightMatrix = []
            for j in range(self.previousChannel):
                currentNodeWeightMatrix.append(weightMatrix[j][i])
            current_node = Dense(currentNodeWeightMatrix)
            self.denseNodes.append(current_node)

    def executeDenseLayer(self, featureMap):
        outputArray = []
        self.featureMap = featureMap
        for currentNode in self.denseNodes:
            for channelNumber, currentChannel in enumerate(self.featureMap):
                outputArray.append(currentNode.get_output(currentChannel, channelNumber))
        
        print(self.outputNode.get_output(outputArray, 0))
        





        
        