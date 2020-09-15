import math

#########################
######## POOLING ########
#########################

# Pooling class
class Pooling:
    ### CONSTRUCTOR ###
    # FilterSize = 2; Stride = 1; Mode = 'max'
    # 2 modes: "MAX", "AVG"
    def __init__(self, filterSize = 2, stride = 1, mode = "MAX"):
        self.__filterSize = filterSize
        self.__stride = stride
        self.__mode = mode

    ### GETTER / SETTER ###
    def getFilter(self):
        return self.__filterSize
    def setFilter(self, filterSize):
        self.__filterSize = filterSize

    def getStride(self):
        return self.__stride
    def setStride(self, stride):
        self.__stride = stride
    
    def getMode(self):
        return self.__mode
    def setMode(self, mode):
        self.__mode = mode
    
    ### POOLING METHODS
    # Return matrix partitioned according to filterSize
    def __partitionInput(self, inputMatrix, startPosition):
        result = []
        for i in range(startPosition[0], (startPosition[0] + self.__filterSize)):
            resultRow = []
            for j in range(startPosition[1], (startPosition[1] + self.__filterSize)):
                resultRow.append(inputMatrix[i][j])
            result.append(resultRow)
        return result

    # Maximize function
    def __maximizeFiltered(self, inputMatrix):
        maxResult = -math.inf
        for i in range(len(inputMatrix)):
            for j in range(len(inputMatrix[0])):
                if inputMatrix[i][j] > maxResult:
                    maxResult = inputMatrix[i][j]
        return maxResult
    
    # Average function
    def __averageFiltered(self, inputMatrix):
        total = 0
        size = len(inputMatrix) * len(inputMatrix[0])
        for i in range(len(inputMatrix)):
            for j in range(len(inputMatrix[0])):
                total += inputMatrix[i][j]
        return (total / size)
    
    # Pooling method
    def pool(self, inputMatrix):
        result = []
        startPosition = [0, 0]

        while (startPosition[0] + self.__filterSize) <= len(inputMatrix):
            resultRow = []

            while (startPosition[1] + self.__filterSize) <= len(inputMatrix[0]):
                partitioned = self.__partitionInput(inputMatrix, startPosition)
                resultCell = None
                if (self.__mode == "MAX"):
                    resultCell = self.__maximizeFiltered(partitioned)
                elif (self.__mode == "AVG"):
                    resultCell = self.__averageFiltered(partitioned)
                resultRow.append(resultCell)
                startPosition[1] += self.__stride
            
            result.append(resultRow)
            startPosition[1] = 0
            startPosition[0] += self.__stride
        
        return result
        


    
