from scipy.special import softmax
import numpy as np

valuearr = np.array([2, 3, 5])
valuearr2 = np.array(np.array([5, 3, 2]))

print(valuearr - valuearr2)
print(np.append(valuearr, 5))
print(valuearr)
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