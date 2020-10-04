from scipy.special import softmax
import numpy as np

np.set_printoptions(precision=2)

x = np.array([50,51,12,0.22,12,1231]) 

m = softmax(x)

print(m.sum(axis=0))

print(m)