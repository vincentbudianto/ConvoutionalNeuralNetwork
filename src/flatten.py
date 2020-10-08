import numpy as np

class FlatteningLayer:
    def __init__(self, size_x = None, size_y = None, size_z = None):
        self.is_initiated = False
        if (size_x is not None and size_y is not None and size_z is not None):
            self.is_initiated = True
            self.size_x = size_x
            self.size_y = size_y
            self.size_z = size_z

    def flatten(self, featuremap):
        if not self.is_initiated:
            self.size_x = featuremap.shape[0]
            self.size_y = featuremap.shape[1]
            self.size_z = featuremap.shape[2]
            self.is_initiated = True
        return(featuremap.flatten())
    
    def unflatten(self, featuremap):
        if not self.is_initiated:
            print("Flattening layer not initiated")
            return None
        return featuremap.reshape((self.size_x, self.size_y, self.size_z))
    
    def calcBackwards(self, d_succ, weight_succ):

        #derivative function
        d_func = self.d_func()

        #add bias
        valuearr = np.array(self.flatArray)
        valuearr = np.append(valuearr, 1)

        phase1 = (np.matmul(d_succ[np.newaxis], weight_succ)) 
        derivoutput = np.array([d_func(x2) for x2 in self.outputs])

        phase2 = np.multiply(phase1, derivoutput)

        newweight = np.array((np.matmul(phase2.reshape(-1,1), valuearr[np.newaxis])))

        if self.cache:
            self.deltaweights = self.deltaweights + newweight
        else:
            self.deltaweights = newweight
            self.cache = True

        return phase2

if __name__ == "__main__":
    print("Testing numpy flatten and unflatten")

    flatten_layer = FlatteningLayer()

    a = np.zeros((2, 3, 4))
    for i in range(len(a)):
        for j in range(len(a[0])):
            for k in range(len(a[0][0])):
                a[i][j][k] = i + j - k
    print(a)

    b = flatten_layer.flatten(a)
    print(b)

    c = flatten_layer.unflatten(b)
    print(c)
