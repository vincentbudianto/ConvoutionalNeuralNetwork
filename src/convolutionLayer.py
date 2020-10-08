from convolution import Convolution
from detector import Detector
from pooling import Pooling
from math import floor, ceil
import numpy as np
import cv2

class ConvolutionLayer:
    """
    Class used to represent Convolution Layer in Convolution Neural Network

    ...

    Attributes
    ----------
    convolution : Object
        Convolution process that processes input matrix into feature map
    detector : Object
        Executes the activation function into feature map to prevent linearity
    pooling: Object
        Pooling process that minimizies the output to a desirable size
    """

    def __init__(self, convolution = [], detector = [], pooling = [], inputs=[], outputs=[], inputMapper = None, connectionMapper = None):
        self.inputs = inputs
        self.inputSize = len(inputs)
        self.outputs = outputs
        self.outputSize = len(outputs)
        self.convolution = convolution
        self.convolutionSize = len(convolution)
        self.detector = detector
        self.detectorSize = len(detector)
        self.pooling = pooling
        self.poolingSize = len(pooling)
        self.inputMapper = inputMapper
        self.connectionMapper = connectionMapper

    ### GETTER / SETTER ###
    def setInputs(self, inputs):
        self.inputs = inputs
    def addInputs(self, newInput):
        self.inputs.append(newInput)
    def getInputs(self):
        return self.inputs

    def setOutputs(self, outputs):
        self.outputs = outputs
    def addOutputs(self, newOutput):
        self.outputs.append(newOutput)
    def getOutputs(self):
        return self.outputs

    def setConvolution(self, convolution):
        self.convolution = convolution
    def getConvolution(self):
        return self.convolution

    def setDetector(self, detector):
        self.detector = detector
    def getDetector(self):
        return self.detector

    def setPooling(self, pooling):
        self.pooling = pooling
    def getPooling(self):
        return self.pooling

    """
    Convolutional layer configurations
    - 3 convolutions
    - 3 detectors
    - 3 poolings
    """
    def setConfigurationDefault(self, batchsize, batchperepoch, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode):
        # Convolution
        convolutionList = []
        for i in range(convFilterCount):
            dummyFilter = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
            convolutionList.append(Convolution(batchsize, batchperepoch, None, convPaddingSize, convFilterSize, convFilterSize, convStrideSize, filters = None))
        self.convolution = convolutionList

        # Detector
        detectorList = []
        for i in range(convFilterCount):
            detectorList.append(Detector(None, detectorMode))
        self.detector = detectorList

        # Pooling
        poolingList = []
        for i in range(convFilterCount):
            poolingList.append(Pooling(poolFilterSize, poolFilterSize, poolStrideSize, poolMode))
        self.pooling = poolingList


    """
    Forward Propagation
    Process the input matrix and sends out output
    """
    def convolutionForward(self):
        result = []
        for i in range(len(self.convolution)):
            # Convolution
            self.convolution[i].setImage(np.copy(self.inputs))
            convolutionResult = self.convolution[i].forward()
            # outputName = "convo" + str(i) + ".jpg"
            # cv2.imwrite(outputName, convolutionResult)

            # Detection
            if (self.detector[i].getBias() == None):
                self.detector[i].setBias(convolutionResult)
            self.detector[i].input = convolutionResult
            detectionResult = self.detector[i].activate()

            # Pooling
            poolingResult = np.array(self.pooling[i].pool(detectionResult))
            result.append(poolingResult)

        self.outputs = np.array(result)

    """
    Backward Propagation
    Process the delta error matrix update delta_weight matrix
    """
    def backward_propagation(self, delta_matrix, learning_rate):
        if len(delta_matrix.shape) == 3:
            c = delta_matrix.shape[0]
            for i in range(c):
                self.backward_node(delta_matrix[i], self.convolution[i], self.detector[i], self.pooling[i], learning_rate)
        else:
            print("Backprop other than 3D is not implemented yet")


    def backward_node(self, delta_matrix, convolution, detector, pooling):
        delta_pooling = pooling.back_propagation(delta_matrix)
        delta_detector = detector.back_propagation(delta_pooling)
        delta_convolution = convolution.back_propagation(delta_detector, learning_rate)
        print('delta_convolution', delta_convolution)

    def updateWeight(self, learning_rate):
        for i in range(len(self.convolution)):
            self.convolution[i].updateWeight(learning_rate)

if __name__ == "__main__":
    print("Testing convolutional layer")

    print("Preparing forward input")
    test_forward_matrix = [[[1, 2, 3, 0], [2, 3, 7, 4], [3, 7, 4, 5], [0, 9, 7, 8]], \
    [[4, 2, 7, 0], [5, 2, 3, 1], [1, 3, 4, 2], [5, 4, 6, 7]], \
    [[9, 3, 6, 5], [2, 2, 4, 4], [1, 3, 7, 4], [2, 3, 1, 1]]]

    test_forward_matrix = np.array(test_forward_matrix)
    print("Input shape:", test_forward_matrix.shape)

    convolution_layer = ConvolutionLayer()
    convolution_layer.setConfigurationDefault(1, 1, convFilterCount=3, convFilterSize=3, convPaddingSize=1, convStrideSize=1, \
    detectorMode='relu', poolFilterSize=2, poolStrideSize=2, poolMode='MAX')
    convolution_layer.setInputs(test_forward_matrix)

    convolution_layer.convolutionForward();

    print("CONVOLUTION LAYER RESULT")
    print(convolution_layer.outputs)
    print("Shape = ", convolution_layer.outputs.shape, "\n")

    print("Preparing backward input")

    test_backward_matrix = [[[1, 0.3], [0.2, 0.4]], [[0.4, 1], [0.5, 0]], [[0.4, 0.3], [0, 0.2]]]
    test_backward_matrix = np.array(test_backward_matrix)
    print("Input shape:", test_backward_matrix.shape)

    convolution_layer.backward_propagation(test_backward_matrix, 0.3)









