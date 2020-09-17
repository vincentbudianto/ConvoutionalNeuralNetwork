from extract import extractImage, createRGBMatrix
from convolutionLayer import ConvolutionLayer
import numpy as np

def test():
    extractedImage = extractImage("testo.jpg", True, 10, 10)
    rgbMatrix = createRGBMatrix(extractedImage[0])
    # print(rgbMatrix)

    print("Extraction clear")

    convolutionLayer = ConvolutionLayer()
    convolutionLayer.setConfigurationDefault()
    convolutionLayer.setInputs(np.array(rgbMatrix))

    convolutionLayer.executeConvolutionLayer()

if __name__ == '__main__':
    test()
    
