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
import copy

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
        self.convolution_layer.setInputs((np.array(extractedImage) / 128) - 1)
        self.convolution_layer.convolutionForward()
        flatArray = self.flattening_layer.flatten(self.convolution_layer.outputs)
        self.dense_layer.executeDenseLayer(flatArray)
        self.output_layer.executeDenseLayer(self.dense_layer.outputs)

        prediction = 0 if self.output_layer.outputs[0] >= self.output_layer.outputs[1] else 1

        ######################################
        #       BACKWARD PROPAGATION         #
        #  Consensus Output Node 0 = Cat     #
        #  Consensus Output Node 1 = Dog     #
        ######################################

        d_out = self.output_layer.calcBackwards(label)
        d_out2 = self.dense_layer.calcBackwards(d_out, self.output_layer.getweight())
        d_flatten = self.flattening_layer.calcBackwards(d_out2, self.dense_layer.getweight())
        d_convolution = self.convolution_layer.backward_propagation(d_flatten)

        return [prediction == label, prediction]

    def update_weight(self, learning_rate, momentum):
        self.output_layer.updateWeight(learning_rate)
        self.dense_layer.updateWeight(learning_rate)
        self.convolution_layer.updateWeight(learning_rate, momentum)
    
    def predict(self, fileName):
        extractedImage = extractImage(fileName, True, self.convInputSize, self.convInputSize)
        extractedImage = np.transpose(extractedImage[0],(2, 0, 1))
        self.convolution_layer.setInputs(np.array(extractedImage))
        self.convolution_layer.convolutionForward()
        flatArray = self.flattening_layer.flatten(self.convolution_layer.outputs)
        self.dense_layer.executeDenseLayer(flatArray)
        self.output_layer.executeDenseLayer(self.dense_layer.outputs)

        prediction = 0 if self.output_layer.outputs[0] >= self.output_layer.outputs[1] else 1
        return prediction
    
    def check_predict(self, fileName, label):
        prediction = self.predict(fileName)
        return prediction == label

    def train(self, directory, label, epoch, learning_rate, momentum, val_data, train_data):
        
        all_true = 0
        datas = np.array_split(train_data, epoch)
        
        for i, data in enumerate(datas): 
            print('Epoch ' + str(i + 1) + '/' + str(epoch))
            print('Training')
            total_true = 0
            random.shuffle(train_data)
            for img in data:
                new_label = 1 if img.split('\\')[2].split('.')[0] == 'dog' else 0
                result = self.train_one(img, new_label)
                if result:
                    total_true += 1
                    all_true += 1
                print("TRAIN DONE : ", img, "; Result:", result)

            self.update_weight(learning_rate, momentum)
            print("Accuracy:", total_true / len(train_data))
            print()

            print('Validating')
            total_true_validate = 0
            random.shuffle(val_data)
            for img in val_data:
                new_label = 1 if img.split('\\')[2].split('.')[0] == 'dog' else 0
                result = self.check_predict(img, new_label)
                if result:
                    total_true_validate += 1
                print("VALIDATE DONE : ", img, "; Result:", result)
            print("Validation Accuracy:", total_true_validate / len(val_data))
            print()

        print("AKURALLLLL")
        print(all_true / len(train_data))
        return(all_true / len(train_data))

    

def kfoldxvalidation(Network, directory, label, epoch, learning_rate, kfold):
    listimg = []
    images = []

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('jpg'):
                images.append(os.path.join(subdir, file))

    random.shuffle(images)

    datas = np.array_split(images, kfold)

    for img in datas:
        listimg.append(list(img))

    #KFOLDLOOP
    KFOLDACC = []
    for countimg, img in enumerate(listimg):

        localNetwork = copy.copy(Network)

        datacopy = listimg[:]
        datacopy.remove(img)

        flattened_datacopy = [y for x in datacopy for y in x]

        accnow = localNetwork.train(directory, label, epoch, learning_rate, img, flattened_datacopy)

        savemodel(localNetwork, str(countimg))

        KFOLDACC.append([accnow, localNetwork])
        
    print(KFOLDACC)


def savemodel(network, iteration):
    file_pi = open('latest_modelV' + iteration +'.obj', 'wb')
    pickle.dump(network, file_pi)

def loadmodel(filename):
    filehandler = open(filename, 'rb')
    object = pickle.load(filehandler)
    return(object)

if __name__ == '__main__':
    curNetwork = Network()
    curNetwork.kfoldxvalidation("test_data", label=0, epoch=10, learning_rate=0.1)
    # curNetwork.initiate_network(100, 2, 3, 2, 1, 'relu', 3, 1, 'AVG')
    # curNetwork.train("test_data\cats", label=0, epoch=10,  learning_rate=0.001)
    # #curNetwork.train_one("src\data\hololive29.jpg", 0)
