import numpy as np

class Convolution:
    def __init__(self, batchsize, batchperepoch, image = None, paddingSize = 2, filterSizeH = 3, filterSizeW = 3, strideSize = 1, filters = None):
        self.paddingSize = paddingSize
        self.strideSize = strideSize
        self.filterSizeH = filterSizeH
        self.filterSizeW = filterSizeW

        self.batchsize = batchsize
        self.batchperepoch = batchperepoch
        self.cache = False
        self.deltafilters = None

        if image is not None:
            self.image = np.transpose(image,(2, 0, 1))

        if filters is None and image is not None:
            h, w, t = image.shape
            self.numFilters = t
            self.filters = np.random.randn(t, filterSizeH, filterSizeW) / (filterSizeH * filterSizeW)
        elif filters is not None:
            self.filters = filters
            self.numFilters = len(filters)
        else:
            self.numFilters = 0

    ### GETTER / SETTER ###
    def getImage(self):
        return self.image
    def setImage(self, image):
        self.image = image

        if image is not None:
            h, w, t = image.shape
            self.filters = np.random.randn(t, self.filterSizeH, self.filterSizeW) / (self.filterSizeH * self.filterSizeW)
            self.numFilters = t

    def getPadding(self):
        return self.paddingSize
    def setPadding(self, paddingSize):
        self.paddingSize = paddingSize

    def getFilterSize(self):
        return self.filterSizeH, self.filterSizeW
    def setFilterSize(self, filterSizeH, filterSizeW):
        self.filterSizeH = filterSizeH
        self.filterSizeW = filterSizeW

    def getStride(self):
        return self.strideSize
    def setStride(self, strideSize):
        self.strideSize = strideSize

    def padding(self):
        result = []

        for imageLayer in self.image:
            tempResult = np.zeros((imageLayer.shape[0] + (self.paddingSize * 2), imageLayer.shape[1] + (self.paddingSize * 2)))

            for i in range(self.paddingSize, (tempResult.shape[0] - self.paddingSize)):
                for j in range(self.paddingSize, (tempResult.shape[1] - self.paddingSize)):
                    tempResult[i, j] = imageLayer[i - self.paddingSize, j - self.paddingSize]

            result.append(tempResult)

        result = np.array(result)

        return result

    def forwardExtract(self, padding):
        h, w = padding.shape

        for i in range(0, (h - (self.filterSizeH - self.strideSize)), self.strideSize):
            for j in range(0, (w - (self.filterSizeW - self.strideSize)), self.strideSize):
                region = padding[i:(i + self.filterSizeH), j:(j + self.filterSizeW)]

                if (region.shape[0] == self.filterSizeH and region.shape[1] == self.filterSizeW):
                    yield region, i, j

    def forward(self):
        self.output = self.padding()
        totalResult = None

        for k in range(len(self.output)):
            paddingLayer = self.output[k]
            result = np.zeros(paddingLayer.shape)

            for curr_region, i, j in self.forwardExtract(paddingLayer):
                curr_result = np.tensordot(curr_region, self.filters[k])
                result[i, j] = np.sum(curr_result)

            output = result[0:result.shape[0] - np.uint16(self.filterSizeH - 1):self.strideSize, 0:result.shape[1] - np.uint16(self.filterSizeW - 1):self.strideSize]

            if totalResult is None:
                totalResult = output
            else:
                totalResult += output

        return totalResult

    def backwardExtract(self, padding, sizeH, sizeW):
        h, w = padding.shape

        for i in range(0, (h - (sizeH - self.strideSize)), self.strideSize):
            for j in range(0, (w - (sizeW - self.strideSize)), self.strideSize):
                region = padding[i:(i + sizeH), j:(j + sizeW)]

                if (region.shape[0] == sizeH and region.shape[1] == sizeW):
                    yield region, i, j

    def back_propagation(self, delta_matrix):
        filters = np.zeros(delta_matrix.shape)
        h, w = filters.shape

        for k in range(len(self.output)):
            paddingLayer = self.output[k]

            for curr_region, i, j in self.backwardExtract(paddingLayer, w, h):
                filters += delta_matrix[i, j] * curr_region

        if self.cache:
            self.deltafilters = self.deltafilters + filters
        else:
            self.deltafilters = filters
            self.cache = True

        print('convolution filters:')
        print(self.deltafilters)

        return self.deltafilters

    def updateFilters(self, learning_rate):
        self.cache = False
        self.filters -= (self.deltafilters / (self.batchsize * self.batchperepoch)) * learning_rate

        print('convolution filters:')
        print(self.filters)
        print(self.deltafilters)

        return self.filters