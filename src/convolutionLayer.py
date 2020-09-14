from convolution import Convolution
from detector import Detector
from pooling import Pooling


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

    def __init__(self, convolution, detector, pooling, inputs=[], outputs=[]):
        self.__inputs = inputs
        self.__outputs = outputs
        self.__convolution = convolution
        self.__detector = detector
        self.__pooling = pooling
        if self.convolution == None:
            self.convolution = Convolution()
        if self.detector == None:
            self.detector = Detector()
        if self.pooling == None:
            self.pooling = Pooling()

    """
    Getter setter untuk setiap atribut
    """

    def setInputs(inputs):
        self.__inputs = inputs
    def addInputs(newInput):
        self.__inputs.append(newInput)
    def getInputs():
        return self.__inputs

    def setOutputs(outputs):
        self.__outputs = outputs
    def addOutputs(newOutput):
        self.__outputs.append(newOutput)
    def getOutputs():
        return self.__outputs

    def setConvolution(convolution):
        self.__convolution = convolution
    def getConvolution():
        return self.__convolution

    def setDetector(detector):
        self.__detector = detector
    def getDetector():
        return self.__detector

    def setPooling(pooling):
        self.__pooling = pooling
    def getPooling():
        return self.__pooling

    """
    Process the input matrix and sends out output
    """
    def executeConvolutionLayer():
        print("test")
        

