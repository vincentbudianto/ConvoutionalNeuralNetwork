from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
from network import Network
import numpy as np
import pickle 

def savemodel(network):
    file_pi = open('latest_model.obj', 'wb') 
    pickle.dump(network, file_pi)

def loadmodel():
    filehandler = open('latest_model.obj', 'rb') 
    object = pickle.load(filehandler)
    return(object)

if __name__ == '__main__':
    curNetwork = Network()
    curNetwork.initiate_network(100, 2, 3, 2, 1, 3, 1, 'AVG')
    curNetwork.train_one("src\data\hololive29.jpg", 0)

    savemodel(curNetwork)
    loadedNetwork = loadmodel() 
    # curNetwork.train("test_data\cats", label=0, epoch=10, batchsize=4, batchperepoch=1, learning_rate=0.001)


