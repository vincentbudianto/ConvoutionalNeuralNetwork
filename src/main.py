from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from flatten import FlatteningLayer
import numpy as np

def test():
    np.set_printoptions(threshold=np.inf)
    extractedImage = extractImage("hololive29.jpg", True, 200, 200)

    print(np.array(extractedImage[0]).shape)

    print("Extraction clear")

    convolutionLayer = ConvolutionLayer()
    convolutionLayer.setConfigurationDefault(3)
    convolutionLayer.setInputs(np.array(extractedImage[0]))

    convolutionLayer.executeConvolutionLayer()

    flatteningLayer = FlatteningLayer()

    flatArray = flatteningLayer.flatten(convolutionLayer.outputs)

    convshape = convolutionLayer.outputs.shape
    print(convolutionLayer.outputs.shape)

    denseLayer = DenseLayer(convshape[0] * convshape[1] * convshape[2])
    denseLayer.initiateLayer()
    denseLayer.executeDenseLayer(flatArray)

if __name__ == '__main__':
    test()
    
