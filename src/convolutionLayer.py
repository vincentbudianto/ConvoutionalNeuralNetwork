from convolution import Convolution
from detector import Detector
from pooling import Pooling
from connectionMapper import ConnectionMapper
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

    """
    Getter setter untuk setiap atribut
    """
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
    def setConfigurationDefault(self, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, poolFilterSize, poolStrideSize, poolMode):
        # Convolution
        convolutionList = []
        for i in range(convFilterCount):
            dummyFilter = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
            convolutionList.append(Convolution(None, convPaddingSize, convFilterSize, convFilterSize, convStrideSize, filters = dummyFilter))
        self.convolution = convolutionList

        # Detector
        detectorList = []
        for i in range(convFilterCount):
            detectorList.append(Detector(None, "relu"))
        self.detector = detectorList

        # Pooling
        poolingList = []
        for i in range(convFilterCount):
            poolingList.append(Pooling(poolFilterSize, poolFilterSize, poolStrideSize, poolMode))
        self.pooling = poolingList

 
    """
    Process the input matrix and sends out output
    """
    def executeConvolutionLayer(self):
        result = []
        self.inputs = np.transpose(self.inputs,(2, 0, 1))
        for i in range(len(self.convolution)):
            # Convolution
            self.convolution[i].setImage(np.copy(self.inputs))
            convolutionResult = self.convolution[i].forward()
            outputName = "convo" + str(i) + ".jpg"
            cv2.imwrite(outputName, convolutionResult)

            # Detection
            if (self.detector[i].getBias() == None):
                self.detector[i].setBias(convolutionResult)
            self.detector[i].input = convolutionResult
            detectionResult = self.detector[i].activate()

            # Pooling
            poolingResult = np.array(self.pooling[i].pool(detectionResult))
            
            result.append(poolingResult)
        
        self.outputs = np.array(result)
        print(self.outputs)





