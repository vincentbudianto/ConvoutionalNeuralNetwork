from scipy.special import softmax
import numpy as np

<<<<<<< HEAD
valuearr = np.array([[[2, 3, 5],[2, 3, 4]],[[2, 3, 1], [2, 3, 3]]])

print(valuearr.shape)
print(valuearr[0].shape)
=======
# valuearr = np.array([2, 3, 5])
# valuearr2 = np.array(np.array([5, 3, 2]))

# print(valuearr - valuearr2)
# print(np.append(valuearr, 5))

>>>>>>> 9a85819c44c9a05c8278a5adee8b36e821cd27da
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

valuearr = np.array([1, 118, 102])
valuearr2 = np.array([1, 1, 0])

# phase1 = np.array([9.855e-01, 1.4199e-02, 2.045e-04, 2.94e-06, 4.24e-08,6.11e-10,8.81e-12,6.11e-10,4.24e-08,-1])
# phase2 = np.array([[0, 0.09, 0.02],[0, 0.08, 0.03],[0, 0.07, 0.03],[0, 0.06, 0.02],[0, 0.05, 0.01],[0, 0.04, 0.02],[0, 0.03, 0.07],[0, 0.04, 0.08],[0, 0.05, 0.05],[0, 0.01, 0.01]])

# phase1 = (np.matmul(phase1[np.newaxis], phase2)) 

# newweight = np.array((np.matmul(phase1.reshape(-1,1), valuearr[np.newaxis])))

# print(phase1[np.newaxis].shape)
# print(phase2.shape)

print(np.multiply(valuearr, valuearr2))