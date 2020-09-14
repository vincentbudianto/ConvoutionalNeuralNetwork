# Just driver for convolution
from convolution import Convolution
import numpy as np

# Defaults
testMatrix = np.arange(25).reshape(5,5)
print("Input")
print(testMatrix)

conv = Convolution()
conv.setImage(testMatrix)

result = conv.forward()
print("\nResult")
print(result)