from extract import extractImage, createRGBMatrix
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
import numpy as np

def test():
    extractedImage = extractImage("testo.jpg", True, 500, 500)
    rgbMatrix = createRGBMatrix(extractedImage[0])
    # print(rgbMatrix)

    print("Extraction clear")

    convolutionLayer = ConvolutionLayer()
    convolutionLayer.setConfigurationDefault(10)
    convolutionLayer.setInputs(np.array(rgbMatrix))

    convolutionLayer.executeConvolutionLayer()

    denseLayer = DenseLayer()
    denseLayer.initiateLayer()
    denseLayer.executeDenseLayer(convolutionLayer.outputs)

if __name__ == '__main__':
    test()
    
