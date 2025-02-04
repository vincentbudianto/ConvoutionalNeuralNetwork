from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
from network import Network, kfoldxvalidation, mass_predict, loadmodel
import numpy as np
import pickle

if __name__ == '__main__':
    curNetwork = Network()
    curNetwork.initiate_network(batchsize = 10, batchperepoch = 24, convInputSize = 16, convFilterCount = 2, convFilterSize = 3, convPaddingSize = 2, convStrideSize = 1, detectorMode = 'relu', poolFilterSize = 3, poolStrideSize = 1, poolMode = 'AVG')
    kfoldxvalidation(curNetwork, "train_data", label=0, epoch=10, learning_rate=0.01, kfold=10, momentum=0.002)
    # curNetwork = loadmodel("latest_modelV9.obj")
    mass_predict(curNetwork, "test_data")
    # curNetwork.kfoldxvalidation("train_data", label=0, epoch=10, learning_rate=0.001, momentum=0.1)

    # test("src\data\hololive29.jpg", 200, 3, 3, 2, 1, 'relu', 3, 1, 'AVG')
    # curNetwork.train("test_data\cats", label=0, epoch=10, batchsize=4, batchperepoch=1, learning_rate=0.001)


