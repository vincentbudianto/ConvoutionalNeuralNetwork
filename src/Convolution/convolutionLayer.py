import extract
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

    def __init__(self, convolution, detector, pooling):
        self.convolution = convolution
        self.detector = detector
        self.pooling = pooling
        if self.convolution == None:
            self.convolution = Convolution()
        if self.detector == None:
            self.detector = Detector()
        if self.pooling == None:
            self.pooling = Pooling()

        
