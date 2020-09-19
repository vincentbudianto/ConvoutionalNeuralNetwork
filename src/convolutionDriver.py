# Just driver for convolution
from extract import extractImage
from convolution import Convolution
import numpy as np


extractedImage = extractImage("hololive29.jpg", True, 200, 200)
extractedImage = np.array(extractedImage[0])
# print(extractedImage)
conv = Convolution(np.array(extractedImage))

result = conv.forward()
print("\nResult")
print(result)