from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
import numpy as np

def test(fileName, convInputSize, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, poolFilterSize, poolStrideSize, poolMode):
    # np.set_printoptions(threshold=np.inf)
    extractedImage = extractImage(fileName, True, convInputSize, convInputSize)

    convolutionLayer = ConvolutionLayer()
    convolutionLayer.setConfigurationDefault(convFilterCount, convFilterSize, convPaddingSize, convStrideSize, poolFilterSize, poolStrideSize, poolMode)
    convolutionLayer.setInputs(np.array(extractedImage[0]))

    convolutionLayer.executeConvolutionLayer()

    print("CONVOLUTION LAYER RESULT")
    print(convolutionLayer.outputs)

    flatteningLayer = FlatteningLayer()

    flatArray = flatteningLayer.flatten(convolutionLayer.outputs)

    convshape = convolutionLayer.outputs.shape
    print(convolutionLayer.outputs.shape)

    denseLayer = DenseLayer(convshape[0] * convshape[1] * convshape[2])
    denseLayer.initiateLayer()
    denseLayer.executeDenseLayer(flatArray)

    dens1shape = denseLayer.outputs.shape

    outputLayer = OutputLayer(dens1shape[0])
    outputLayer.initiateLayer()
    outputLayer.executeDenseLayer(denseLayer.outputs)
    print("OUTPUT RESULT")
    print(outputLayer.outputs)

if __name__ == '__main__':
    test("soberu.png", 200, 2, 3, 2, 1, 3, 1, 'AVG')

