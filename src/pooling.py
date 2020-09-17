import math

# Pooling class
class Pooling:
    ### CONSTRUCTOR ###
    # filterWidth = filterHeight = 2; Stride = 1; Mode = 'max'
    # 2 modes: "MAX", "AVG"
    def __init__(self, filterWidth = 2, filterHeight = 2, stride = 1, mode = 'MAX'):
        self.__filterWidth = filterWidth
        self.__filterHeight = filterHeight
        self.__filterHeight = filterHeight
        self.__stride = stride
        self.__mode = mode

    ### GETTER / SETTER ###
    def getFilterWidth(self):
        return self.__filterWidth
    def setFilterWidth(self, filterWidth):
        self.__filterWidth = filterWidth
    
    def getFilterHeight(self):
        return self.__filterHeight
    def setFilterHeight(self, filterHeight):
        self.__filterHeight = filterHeight

    def getStride(self):
        return self.__stride
    def setStride(self, stride):
        self.__stride = stride
    
    def getMode(self):
        return self.__mode
    def setMode(self, mode):
        self.__mode = mode
    
    ### POOLING METHODS
    # Return matrix partitioned according to filterWidth and filterHeight
    def __partitionInput(self, inputMatrix, startPosition):
        result = []
        for i in range(startPosition[0], (startPosition[0] + self.__filterHeight)):
            resultRow = []
            for j in range(startPosition[1], (startPosition[1] + self.__filterWidth)):
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

        while (startPosition[0] + self.__filterHeight) <= len(inputMatrix):
            resultRow = []

            while (startPosition[1] + self.__filterWidth) <= len(inputMatrix[0]):
                partitioned = self.__partitionInput(inputMatrix, startPosition)
                resultCell = None
                if (self.__mode == 'MAX'):
                    resultCell = self.__maximizeFiltered(partitioned)
                elif (self.__mode == 'AVG'):
                    resultCell = self.__averageFiltered(partitioned)
                resultRow.append(resultCell)
                startPosition[1] += self.__stride
            
            result.append(resultRow)
            startPosition[1] = 0
            startPosition[0] += self.__stride
        
        return result
        


    
