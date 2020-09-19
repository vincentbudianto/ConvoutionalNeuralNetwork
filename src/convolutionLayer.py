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
    def setConfigurationDefault(self, kernelSize):
        # Convolution
        convolutionList = []
        for i in range(1, 4):
            #kernel = np.arange(i, kernelSize * kernelSize * i + 1, i).reshape(kernelSize,kernelSize)
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            convolutionList.append(Convolution(None, paddingSize = 0, filterCount = 1, filterSizeH = kernelSize, filterSizeW = kernelSize, strideSize = 1, filters = kernel))
        self.convolution = convolutionList

        # Detector
        detectorList = []
        detectorList.append(Detector(None, "relu"))
        detectorList.append(Detector(None, "relu"))
        detectorList.append(Detector(None, "relu"))
        #detectorList.append(Detector(None, "sigmoid"))
        self.detector = detectorList

        # Pooling
        poolingList = []
        poolingList.append(Pooling(2, 2, 2, "AVG"))
        poolingList.append(Pooling(2, 2, 2, "AVG"))
        poolingList.append(Pooling(2, 2, 2, "AVG"))
        #poolingList.append(Pooling(3, 3, 1, "AVG"))
        self.pooling = poolingList

        # Input Mapper
        inputMapper = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        h, w = inputMapper.shape
        inputMapper = ConnectionMapper(h, w, inputMapper)
        self.inputMapper = inputMapper

        # Connection Mapper
        connectionMapper = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        h, w = connectionMapper.shape
        connectionMapper = ConnectionMapper(h, w, connectionMapper)
        self.connectionMapper = connectionMapper

 
    """
    Process the input matrix and sends out output
    """
    def executeConvolutionLayer(self):
        # Convolution
        convolutionResult = []
        self.inputs = np.transpose(self.inputs,(2, 0, 1))
        for i in range(len(self.inputs)):
            targetedConvolution = self.inputMapper.getConnectionFromPreviousNode(i)
            for j in range(len(targetedConvolution)):
                if targetedConvolution[j] == 1:
                    self.convolution[j].setImage(np.copy(self.inputs[i]))
        for i in range(len(self.convolution)):
            convolutionResult.append(self.convolution[i].forward())
        
        # cv2.imwrite("convo.jpg", np.transpose(convolutionResult,(1, 2, 0)))

        # print("CONVOLUTION RESULT:\n", convolutionResult)

        # Detection
        detectionResult = []
        for i in range(self.connectionMapper.getNextNodeCount()):
            targetedDetection = self.connectionMapper.getConnectionFromNextNode(i)
            detection = []
            for j in range(len(convolutionResult)):
                if targetedDetection[j] == 1:
                    if len(detection) == 0:
                        detection = np.zeros(convolutionResult[j].shape)
                    detection += convolutionResult[j]
            if (self.detector[i].getBias() == None):
                self.detector[i].setBias(detection)
                # print('set bias')
            detectionResult.append(self.detector[i].forward_activation(detection))

        #print("DETECTION RESULT:\n",detectionResult)

        # Pooling
        result = []
        for i in range(len(detectionResult)):
            result.append(np.array(self.pooling[i].pool(detectionResult[i])))
        
        self.outputs = np.array(result)
        self.outputSize = len(np.array(result))
        
        self.outputs = np.transpose(self.outputs,(1, 2, 0))

        # cv2.imwrite("finalconvo.jpg", self.outputs)

        # print("RESULT:\n", self.outputs)
        # print("SHAPE:\n", self.outputs.shape)





