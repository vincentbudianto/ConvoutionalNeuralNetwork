from convolution import Convolution
from detector import Detector
from pooling import Pooling
from math import floor, ceil
import numpy as np


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

    def __init__(self, convolution = [], detectors = [], pooling = [], inputs=[], outputs=[]):
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
    
    """
    Getter setter untuk setiap atribut
    """
    def setInputs(inputs):
        self.inputs = inputs
    def addInputs(newInput):
        self.inputs.append(newInput)
    def getInputs():
        return self.inputs

    def setOutputs(outputs):
        self.outputs = outputs
    def addOutputs(newOutput):
        self.outputs.append(newOutput)
    def getOutputs():
        return self.outputs

    def setConvolution(convolution):
        self.convolution = convolution
    def getConvolution():
        return self.convolution

    def setDetector(detector):
        self.detector = detector
    def getDetector():
        return self.detector

    def setPooling(pooling):
        self.pooling = pooling
    def getPooling():
        return self.pooling
    
    """
    Convolutional layer configurations
    - 6 convolutions
    - 2 detectors
    - 2 poolings
    """
    def setConfigurationDefault():
        # Convolution
        convolutionList = []
        for i in range(1, 7):
            kernel = np.arrange(i, 25 * i, i).reshape(5,5)
            convolutionList.append(Convolution(None, 1, 1, 5, 5, 1, kernel))
        self.convolution = convolutionList

        # Detector
        detectorList = []
        detectorList.append(Detector(None, "relu"))
        detectorList.append(Detector(None, "sigmoid"))
        self.detector = detectorList

        # Pooling
        poolingList = []
        poolingList.append(Pooling(2, 2, "MAX"))
        poolingList.append(Pooling(3, 1, "AVG"))
        self.pooling = poolingList


    """
    Process the input matrix and sends out output
    """
    def executeConvolutionLayer():
        print("test")
        

