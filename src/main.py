from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
from network import Network
import numpy as np
import pickle

# def test(fileName, convInputSize, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode):
    # # np.set_printoptions(threshold=np.inf)
    # extractedImage = extractImage(fileName, True, convInputSize, convInputSize)
    # extractedImage = np.transpose(extractedImage[0],(2, 0, 1))

    # convolutionLayer = ConvolutionLayer()
    # convolutionLayer.setConfigurationDefault(convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode)
    # convolutionLayer.setInputs(np.array(extractedImage))

    # convolutionLayer.convolutionForward()

    # print("CONVOLUTION LAYER RESULT")
    # print(convolutionLayer.outputs)

    # flatteningLayer = FlatteningLayer()

    # flatArray = flatteningLayer.flatten(convolutionLayer.outputs)

    # convshape = convolutionLayer.outputs.shape
    # print(convolutionLayer.outputs.shape)

    # denseLayer = DenseLayer(convshape[0] * convshape[1] * convshape[2], 1, 1, "relu")
    # denseLayer.initiateLayer()
    # denseLayer.executeDenseLayer(flatArray)

    # dens1shape = denseLayer.outputs.shape

    # outputLayer = OutputLayer(dens1shape[0], 1, 1)
    # outputLayer.initiateLayer()
    # outputLayer.executeDenseLayer(denseLayer.outputs)
    # print("OUTPUT RESULT")
    # print(outputLayer.outputs)

    # #BACKWARD PROPAGATION
    # #Consensus Output Node 0 = Cat
    # #Consensus Output Node 1 = Dog
    # # error = outputLayer.computeError(0)
    # d_out = outputLayer.calcBackwards(0)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(1)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(1)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(0)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(1)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(1)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(0)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(0)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(1)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # d_out = outputLayer.calcBackwards(0)
    # d_out2 = denseLayer.calcBackwards(d_out, outputLayer.getweight())
    # outputLayer.updateWeight(0.001)
    # denseLayer.updateWeight(0.001)

    # print("METADATA")
    # print("D_OUT2")
    # print(d_out2)
    # print("WEIGHT")
    # print(denseLayer.getweight())
    # print(denseLayer.getweight().shape)
    # print("FLATLENGTH")
    # print(denseLayer.flatlength)

def savemodel(network):
    file_pi = open('latest_model.obj', 'wb')
    pickle.dump(network, file_pi)

def loadmodel():
    filehandler = open('latest_model.obj', 'rb')
    object = pickle.load(filehandler)
    return(object)

if __name__ == '__main__':
    curNetwork = Network()
    curNetwork.initiate_network(batchsize = 2, batchperepoch = 9, convInputSize = 100, convFilterCount = 2, convFilterSize = 3, convPaddingSize = 2, convStrideSize = 1, detectorMode = 'relu', poolFilterSize = 3, poolStrideSize = 1, poolMode = 'AVG')
    curNetwork.kfoldxvalidation("test_data", label=0, epoch=10, learning_rate=0.001)
    # curNetwork.kfoldxvalidation("train_data", label=0, epoch=10, learning_rate=0.001)

    savemodel(curNetwork)
    loadedNetwork = loadmodel()
    # test("src\data\hololive29.jpg", 200, 3, 3, 2, 1, 'relu', 3, 1, 'AVG')
    # curNetwork.train("test_data\cats", label=0, epoch=10, batchsize=4, batchperepoch=1, learning_rate=0.001)


