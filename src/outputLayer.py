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
            current_node = Dense(currentNodeWeightMatrix, activation_function="softmax")
            self.denseNodes.append(current_node)

    def executeDenseLayer(self, flatArray):
        outputArray = np.array([])
        self.flatArray = flatArray
        for currentNode in self.denseNodes:
            outputArray = np.append(outputArray, currentNode.get_output(flatArray))

        self.outputs = softmax(outputArray)


    def computeError(self, label):
        if label == 0:
            #cat
            return -np.log(self.outputs[0])

        elif label == 1:
            #dog
            return -np.log(self.outputs[1])

    def updateWeight(self, label, learningrate):
        phase1 = []
        newweight = []

        #add bias
        valuearr = self.flatArray
        valuearr.append(1)

        for x, output in enumerate(self.outputs):
            phase1.append(output if (x != label) else (-1 * (1 - output)))

        #nodecount + bias

        for val in valuearr:
            newweight.append(np.dot(phase1, val).tolist())

        for i, node in enumerate(self.denseNodes):
            nodenewweight = []
            for weight in newweight:
                nodenewweight.append(weight[i])
            node.updateWeight(nodenewweight, learningrate)








