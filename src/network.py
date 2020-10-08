from extract import extractImage
from convolutionLayer import ConvolutionLayer
from denseLayer import DenseLayer
from outputLayer import OutputLayer
from flatten import FlatteningLayer
import numpy as np
import pandas as pd
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
        self.batchsize = batchsize
        self.batchperepoch = batchperepoch
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
        print("Backwards convolution:", d_convolution)

    def update_weight(self, learning_rate):
        self.output_layer.updateWeight(learning_rate)
        self.dense_layer.updateWeight(learning_rate)
        self.convolution_layer.updateWeight(learning_rate)

    def train(self, directory, label, epoch, learning_rate, val_data, train_data):
        for i in range(epoch):
            random.shuffle(train_data)
            imglist = train_data

            for j in range(self.batchperepoch):
                data = []

                for _ in range(self.batchsize):
                    if (len(imglist) > 0):
                        data.append(imglist.pop(0))

                print('epoch ' + str(i) + '/' + str(epoch) + ' : batch ' + str(j) + '/' + str(self.batchperepoch))

                for img in data:
                    self.train_one(img, label)
                    print("TRAIN DONE :", img)

            self.update_weight(learning_rate)

    def kfoldxvalidation(self, directory, label, epoch, learning_rate):
        listimg = []
        images = []

        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('jpg'):
                    images.append(os.path.join.(subdir, file))

        datas = np.array_split(images, epoch)

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
    curNetwork.kfoldxvalidation("test_data", label=0, epoch=10, learning_rate=0.001)
    # curNetwork.initiate_network(100, 2, 3, 2, 1, 'relu', 3, 1, 'AVG')
    # curNetwork.train("test_data\cats", label=0, epoch=10,  learning_rate=0.001)
    # #curNetwork.train_one("src\data\hololive29.jpg", 0)
