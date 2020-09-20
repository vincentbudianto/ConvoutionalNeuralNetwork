from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from flatten import FlatteningLayer
import numpy as np

def test(convInputSize, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, poolFilterSize, poolStrideSize, poolMode):
    # np.set_printoptions(threshold=np.inf)
    extractedImage = extractImage("hololive29.jpg", True, convInputSize, convInputSize)

    convolutionLayer = ConvolutionLayer()
    convolutionLayer.setConfigurationDefault(convFilterCount, convFilterSize, convPaddingSize, convStrideSize, poolFilterSize, poolStrideSize, poolMode)
    convolutionLayer.setInputs(np.array(extractedImage[0]))

    convolutionLayer.executeConvolutionLayer()

    print("CONVOLUTION DONE")
    print(convolutionLayer.outputs.shape)

    flatteningLayer = FlatteningLayer()

    flatArray = flatteningLayer.flatten(convolutionLayer.outputs)

    convshape = convolutionLayer.outputs.shape
    print(convolutionLayer.outputs.shape)

    denseLayer = DenseLayer(convshape[0] * convshape[1] * convshape[2])
    denseLayer.initiateLayer()
    denseLayer.executeDenseLayer(flatArray)

if __name__ == '__main__':
    test(200, 2, 3, 2, 1, 3, 1, 'AVG')

