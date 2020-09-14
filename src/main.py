from extract import extractImage, createRGBMatrix
from convolutionLayer import ConvolutionLayer

if __name__ == '__main__':
    extractedImage = extractImage("testo.jpg", True, 10, 10)
    rgbMatrix = createRGBMatrix(extractedImage[0])
    print("Red =", rgbMatrix[0])
    print("Green =", rgbMatrix[1])
    print(extractedImage.shape)
