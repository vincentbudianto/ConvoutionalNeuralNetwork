# Just driver for detector
from detector import Detector
import numpy as np

testMatrix = np.arange(25).reshape(5,5) - np.ones(25).reshape(5,5) * 15
print("Input")
print(testMatrix)

det = Detector(input=testMatrix)

result = det.activate()
print("\nResult")
print(result)