import numpy as np
from dense import Dense
from scipy.special import softmax

class OutputLayer:
    def __init__(self, flatlength, batchsize, batchperepoch, nodeCount = 2):
        self.flatArray = None
        self.flatlength = flatlength
        self.denseNodes = []
        self.nodeCount = nodeCount
        self.outputs = None
        self.deltaweights = None
        self.cache = False
        self.batchsize = batchsize
        self.batchperepoch = batchperepoch

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

    def calcBackwards(self, label):
        phase1 = np.array([])

        #add bias
        valuearr = np.array(self.flatArray)
        valuearr = np.append(valuearr, 1)

        for x, output in enumerate(self.outputs):
            phase1 = np.append(phase1 , (output if (x != label) else (-1 * (1 - output))))

        newweight = np.array((np.matmul(phase1.reshape(-1,1), valuearr[np.newaxis])))

        if self.cache:
            self.deltaweights = self.deltaweights + newweight
        else:
            self.deltaweights = newweight
            self.cache = True

    def updateWeight(self, learningrate):

        self.cache = False

        self.deltaweights = (self.deltaweights / (self.batchsize * self.batchperepoch)) * learningrate

        for i, node in enumerate(self.denseNodes):
            node.updateWeight(self.deltaweights[i])









