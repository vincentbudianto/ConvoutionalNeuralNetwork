from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
import numpy as np

def test(fileName, convInputSize, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode):
    # np.set_printoptions(threshold=np.inf)
    extractedImage = extractImage(fileName, True, convInputSize, convInputSize)
    extractedImage = np.transpose(extractedImage[0],(2, 0, 1))

    convolutionLayer = ConvolutionLayer()
    convolutionLayer.setConfigurationDefault(convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode)
    convolutionLayer.setInputs(np.array(extractedImage))

    convolutionLayer.convolutionForward()

    print("CONVOLUTION LAYER RESULT")
    print(convolutionLayer.outputs)

    flatteningLayer = FlatteningLayer()

    flatArray = flatteningLayer.flatten(convolutionLayer.outputs)

    convshape = convolutionLayer.outputs.shape
    print(convolutionLayer.outputs.shape)

    denseLayer = DenseLayer(convshape[0] * convshape[1] * convshape[2], 1, 1, "relu")
    denseLayer.initiateLayer()
    denseLayer.executeDenseLayer(flatArray)

    dens1shape = denseLayer.outputs.shape

    outputLayer = OutputLayer(dens1shape[0], 1, 1)
    outputLayer.initiateLayer()
    outputLayer.executeDenseLayer(denseLayer.outputs)
    print("OUTPUT RESULT")
    print(outputLayer.outputs)

    #BACKWARD PROPAGATION
    #Consensus Output Node 0 = Cat
    #Consensus Output Node 1 = Dog
    # error = outputLayer.computeError(0)
    d_out = outputLayer.calcBackwards(0)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(1)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(1)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(0)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(1)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(1)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(0)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(0)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(1)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    d_out = outputLayer.calcBackwards(0)
    d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    outputLayer.updateWeight(0.001)
    denseLayer.updateWeight(0.001)

    print("METADATA")
    print("D_OUT2")
    print(d_out2)
    print("WEIGHT")
    print(denseLayer.getweight())
    print(denseLayer.getweight().shape)
    print("FLATLENGTH")
    print(denseLayer.flatlength)

if __name__ == '__main__':
    test("src\data\hololive29.jpg", 200, 3, 3, 2, 1, 'relu', 3, 1, 'AVG')


