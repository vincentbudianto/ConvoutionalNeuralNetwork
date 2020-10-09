import numpy as np
from dense import Dense



class DenseLayer:
    def __init__(self, flatlength, batchsize, batchperepoch, activation_function="relu", nodeCount = 10):
        self.flatArray = None
        self.flatlength = flatlength
        self.denseNodes = []
        self.nodeCount = nodeCount
        self.outputs = None
        self.deltaweights = None
        self.cache = False
        self.batchsize = batchsize
        self.batchperepoch = batchperepoch
        self.activation_function = activation_function
        self.previousweights = None

    def initiateLayer(self):
        for _ in range(self.nodeCount):
            currentNodeWeightMatrix = (np.random.randn(self.flatlength) % 2) - 1
            current_node = Dense(currentNodeWeightMatrix, activation_function = self.activation_function)
            self.denseNodes.append(current_node)

    def executeDenseLayer(self, flatArray):
        outputArray = np.array([])
        self.flatArray = flatArray
        for currentNode in self.denseNodes:
            outputArray = np.append(outputArray, currentNode.get_output(flatArray))

        self.outputs = outputArray


    def d_func(self):
        if self.activation_function == "sigmoid":
            return lambda x : x * (1 - x)
        elif self.activation_function == "relu":
            return lambda x : 1 * (x > 0)

    def calcBackwards(self, d_succ, weight_succ):

        #derivative function
        d_func = self.d_func()

        #add bias
        valuearr = np.array(self.flatArray)
        valuearr = np.append(valuearr, 1)

        phase1 = (np.matmul(d_succ[np.newaxis], weight_succ)) 
        derivoutput = np.array([d_func(x2) for x2 in self.outputs])

        phase2 = np.multiply(phase1, derivoutput)

        newweight = np.array((np.matmul(phase2.reshape(-1,1), valuearr[np.newaxis])))

        if self.cache:
            self.deltaweights = self.deltaweights + newweight
        else:
            self.deltaweights = newweight
            self.cache = True

        return phase2


    def updateWeight(self, learningrate, momentum):

        self.cache = False

        self.deltaweights = (self.deltaweights / (self.batchsize * self.batchperepoch)) * learningrate + ((momentum * self.previousweights) if self.previousweights is not None else 0)

        self.previousweights = self.deltaweights

        for i, node in enumerate(self.denseNodes):
            node.updateWeight(self.deltaweights[i])


    def getweight(self):

        allweight = []

        for nodes in self.denseNodes:
            allweight.append(nodes.get_weight())
        
        return np.array(allweight)





