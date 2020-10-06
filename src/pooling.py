import math
import numpy as np

# Pooling class
class Pooling:
    ### CONSTRUCTOR ###
    # filterWidth = filterHeight = 2; Stride = 1; Mode = 'max'
    # 2 modes: "MAX", "AVG"
    def __init__(self, filterWidth = 2, filterHeight = 2, stride = 2, mode = 'MAX'):
        self.__filterWidth = filterWidth
        self.__filterHeight = filterHeight
        self.__filterHeight = filterHeight
        self.__stride = stride
        self.__mode = mode
        self.shape_initialized = False
        self.shape_x = None
        self.shape_y = None
        self.max_location = []

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
        idx_row = -1
        idx_col = -1
        for i in range(len(inputMatrix)):
            for j in range(len(inputMatrix[0])):
                if inputMatrix[i][j] > maxResult:
                    maxResult = inputMatrix[i][j]
                    idx_row = i
                    idx_col = j

        return (maxResult, idx_row, idx_col)

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
        max_location = []

        if not self.shape_initialized:
            self.shape_x = len(inputMatrix)
            self.shape_y = len(inputMatrix[0])
            self.shape_initialized = True

        while (startPosition[0] + self.__filterHeight) <= len(inputMatrix):
            resultRow = []
            max_row = []

            while (startPosition[1] + self.__filterWidth) <= len(inputMatrix[0]):
                partitioned = self.__partitionInput(inputMatrix, startPosition)
                resultCell = None
                if (self.__mode == 'MAX'):
                    maxResult = self.__maximizeFiltered(partitioned)
                    resultCell = maxResult[0]
                    max_row.append((maxResult[1], maxResult[2]))
                elif (self.__mode == 'AVG'):
                    resultCell = self.__averageFiltered(partitioned)
                resultRow.append(resultCell)
                startPosition[1] += self.__stride

            result.append(resultRow)
            max_location.append(max_row)
            startPosition[1] = 0
            startPosition[0] += self.__stride

        self.max_location = max_location
        return result
    
    # Back propagation
    def back_propagation(self, delta_matrix):
        result = np.zeros((self.shape_x, self.shape_y))

        startPosition = [0, 0]
        i = 0
        while (startPosition[0] + self.__filterHeight) <= self.shape_x:
            j = 0
            while (startPosition[1] + self.__filterWidth) <= self.shape_y:
                print(i, j)
                delta_pos = delta_matrix[i][j]
                partitioned = result[startPosition[0] : startPosition[0] + self.__filterHeight, startPosition[1] : startPosition[1] + self.__filterWidth]
                if (self.__mode == 'MAX'):
                    max_pos = self.max_location[i][j]
                    partitioned[max_pos[0]][max_pos[1]] += delta_pos
                elif (self.__mode == 'AVG'):
                    partitioned += delta_pos / (self.__filterHeight * self.__filterWidth)
                startPosition[1] += self.__stride
                j += 1
            startPosition[1] = 0
            startPosition[0] += self.__stride
            i += 1

        return result

if __name__ == "__main__":
    print("Testing pooling layer")

    print("Forward")
    pool_layer = Pooling()
    pool_layer.setMode('AVG')
    test_matrix = np.array([[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]])

    print("Input")
    print(test_matrix)

    result = pool_layer.pool(test_matrix)
    max_result = pool_layer.max_location

    print("Output")
    print(result)
    print("Location")
    print(max_result)

    print("Backward")
    delta_error = np.array([[1, 1], [2, 2]])

    print("Error")
    print(delta_error)

    unpooled_error = pool_layer.back_propagation(delta_error)
    print("Result")
    print(unpooled_error)








