from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
import numpy as np
import pickle
import os
import random

class Network:
    def __init__(self):
        self.convolution_layer = None
        self.flattening_layer = None
        self.dense_layer = None
        self.output_layer = None

        self.convInputSize = None

    def initiate_network(self, batchsize, batchperepoch, convInputSize, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode, activation_dense="relu"):

        self.convInputSize = convInputSize

        self.convolution_layer = ConvolutionLayer()
        self.convolution_layer.setConfigurationDefault(batchsize, batchperepoch, convFilterCount, convFilterSize, convPaddingSize, convStrideSize, detectorMode, poolFilterSize, poolStrideSize, poolMode)

        self.flattening_layer = FlatteningLayer()

        self.dense_layer = DenseLayer(convInputSize * convInputSize * convFilterCount, batchsize, batchperepoch, activation_dense)
        self.dense_layer.initiateLayer()

        self.output_layer = OutputLayer(10, 1, 1)
        self.output_layer.initiateLayer()

    def train_one(self, fileName, label):
        ####################################
        # IMAGE LOAD & FORWARD PROPAGATION #
        ####################################
        extractedImage = extractImage(fileName, True, self.convInputSize, self.convInputSize)
        extractedImage = np.transpose(extractedImage[0],(2, 0, 1))
        self.convolution_layer.setInputs(np.array(extractedImage))
        self.convolution_layer.convolutionForward()

        print("Forward convolution:", self.convolution_layer.outputs.shape)

        flatArray = self.flattening_layer.flatten(self.convolution_layer.outputs)
        print("Forward flatten:", flatArray.shape)

        self.dense_layer.executeDenseLayer(flatArray)
        print("Forward dense:", self.dense_layer.outputs.shape)

        self.output_layer.executeDenseLayer(self.dense_layer.outputs)

        ######################################
        #       BACKWARD PROPAGATION         #
        #  Consensus Output Node 0 = Cat     #
        #  Consensus Output Node 1 = Dog     #
        ######################################

        d_out = self.output_layer.calcBackwards(label)
        print(d_out)
        d_out2 = self.dense_layer.calcBackwards(d_out, self.output_layer.getweight())
        print("Backwards dense:", d_out2.shape)
        d_flatten = self.flattening_layer.calcBackwards(d_out2, self.dense_layer.getweight())
        print("Backwards flatten:", d_flatten.shape)

        d_convolution = self.convolution_layer.backward_propagation(d_flatten)
        print("Backwards convolution:")

    def update_weight(self, learning_rate):
        self.output_layer.updateWeight(learning_rate)
        self.dense_layer.updateWeight(learning_rate)
        self.convolution_layer.updateWeight(learning_rate)

    def train(self, directory, label, epoch, learning_rate, val_data, train_data):

        for _ in range(epoch):
            for img in train_data:
                self.train_one(img, label)
                print("TRAIN DONE : ", img)
            self.update_weight(learning_rate)

    def kfoldxvalidation(self, directory, label, epoch, learning_rate):

        listimg = []

        if directory:
            os.chdir(directory)

        files = os.listdir()

        images = ([file for file in files if file.endswith(('jpg'))])


        datas = (np.array_split(images, epoch))
    
        for img in datas:
            listimg.append(list(img))

        #KFOLDLOOP
        for img in listimg:
            datacopy = listimg[:]
            datacopy.remove(img)

            flattened_datacopy = [y for x in datacopy for y in x]

            self.train(directory, label, epoch, learning_rate, img, flattened_datacopy)




if __name__ == '__main__':
    curNetwork = Network()
<<<<<<< HEAD
    curNetwork.kfoldxvalidation("test_data\cats", label=0, epoch=10, learning_rate=0.001)
    # curNetwork.initiate_network(100, 2, 3, 2, 1, 'relu', 3, 1, 'AVG')
    # curNetwork.train("test_data\cats", label=0, epoch=10,  learning_rate=0.001)
    # #curNetwork.train_one("src\data\hololive29.jpg", 0)
=======
    curNetwork.initiate_network(1, 1, 100, 2, 3, 2, 1, 3, 1, 'AVG')
    curNetwork.train("test_data\cats", label=0, epoch=10, batchsize=4, batchperepoch=1, learning_rate=0.001)
    #curNetwork.train_one("src\data\hololive29.jpg", 0)
>>>>>>> 285b95f83221dae6d799c799c4d9985cf41213a3
