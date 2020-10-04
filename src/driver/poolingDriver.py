# Just driver for pooling
from pooling import Pooling

# Defaults
testMatrix = [[1, 1, 2, 4], [5, 6, 7, 8], [3, 2, 1, 0], [1, 2, 3, 4]]
print("Input")
print(testMatrix)

pooler = Pooling()
pooler.setStride(2)

result = pooler.pool(testMatrix)
print("\nResult")
print(result)

