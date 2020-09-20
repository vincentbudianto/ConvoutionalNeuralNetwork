import numpy as np

class ConnectionMapper:
    """
    Class used to represent connection for a non fully connected layer

    Connects outputs of a layer to inputs of the next layer
    By matrix
    """

    def __init__(self, previousNodeCount = 2, nextNodeCount = 2, connectionMap = None):
        self.previousNodeCount = previousNodeCount
        self.nextNodeCount = nextNodeCount
        if connectionMap is None:
            self.connectionMap = np.zeros(previousNodeCount, nextNodeCount)
        else:
            self.connectionMap = connectionMap

    """
    GETTER / SETTER
    """
    def getConnectionMap(self):
        return self.connectionMap
    def setConnectionMap(self, connectionMap):
        self.connectionMap = connectionMap
        self.previousNodeCount = len(connectionMap)
        self.nextNodeCount = len(connectionMap[0])

    def getPreviousNodeCount(self):
        return self.previousNodeCount
    def setPreviousNodeCount(self, previousNodeCount):
        self.previousNodeCount = previousNodeCount

    def getNextNodeCount(self):
        return self.nextNodeCount
    def setNextNodeCount(self, nextNodeCount):
        self.nextNodeCount = nextNodeCount

    """
    CONNECTION ANALYSIS
    """
    def setConnectionNode(self, previousNode, nextNode):
        self.connectionMap[previousNode, nextNode] = 1
    def unsetConnectionNode(self, previousNode, nextNode):
        self.connectionMap[previousNode, nextNode] = 0

    def getConnectionFromNode(self, previousNode, nextNode):
        return self.connectionMap[previousNode, nextNode]
    def getConnectionFromPreviousNode(self, previousNode):
        return self.connectionMap[previousNode]
    def getConnectionFromNextNode(self, nextNode):
        return [row[nextNode] for row in self.connectionMap]


