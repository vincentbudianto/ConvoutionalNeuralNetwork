from scipy.special import softmax
import numpy as np

valuearr = np.array([[[2, 3, 5],[2, 3, 4]],[[2, 3, 1], [2, 3, 3]]])

print(valuearr.shape)
print(valuearr[0].shape)
# newweight = []
# deltaweights = None
# popog = []

# valuearr = np.array([1, 424, 0])

# phase1 = np.array([9.85e-01, 1.41e-02, 2.04e-04])

# newweight = (np.matmul(phase1.reshape(-1,1), valuearr[np.newaxis]))

# print(newweight)
# deltaweights = newweight + newweight
# print(deltaweights)
# print(deltaweights / 2)