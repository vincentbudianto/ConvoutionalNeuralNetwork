import numpy as np

class Convolution:
  def __init__(self, image = None, paddingSize = 2, filterSizeH = 3, filterSizeW = 3, strideSize = 1, filters = None):
    self.paddingSize = paddingSize
    self.strideSize = strideSize
    self.filterSizeH = filterSizeH
    self.filterSizeW = filterSizeW

    if image is not None:
        self.image = np.transpose(image,(2, 0, 1))
    
    if filters is None and image is not None:
        h, w, t = image.shape
        self.filters = np.random.randn(t, self.filterSizeH, self.filterSizeW) / (self.filterSizeH * self.filterSizeW)
    else:
        self.filters = filters

  ### GETTER / SETTER ###
  def getImage(self):
      return self.image
  def setImage(self, image):
      self.image = image
      if image is not None:
        h, w, t = image.shape
        self.filters = np.random.randn(t, self.filterSizeH, self.filterSizeW) / (self.filterSizeH * self.filterSizeW)

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

  def extract(self, padding):
    '''
    Generates all possible i x j image regions using padding.
    - image is a 2d numpy array.
    '''
    h, w = padding.shape

    for i in range(0, (h - (self.filterSizeH - self.strideSize)), self.strideSize):
        for j in range(0, (w - (self.filterSizeW - self.strideSize)), self.strideSize):
            region = padding[i:(i + self.filterSizeH), j:(j + self.filterSizeW)]

            yield region, i, j

  def forward(self):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w).
    '''
    padding = self.padding()

    totalResult = None

    for k in range(len(padding)):
        paddingLayer = padding[k]
        result = np.zeros(paddingLayer.shape)
        for curr_region, i, j in self.extract(paddingLayer):
            curr_result = np.tensordot(curr_region, self.filters[k])
            result[i, j] = np.sum(curr_result)
            #print(i, j, ':',"CURREGION : \n", curr_region, "FILTER :\n", self.filters, np.sum(curr_result))

        output = result[0:result.shape[0] - np.uint16(self.filterSizeH - 1):self.strideSize, 0:result.shape[1] - np.uint16(self.filterSizeW - 1):self.strideSize]

        if totalResult is None:
            totalResult = output
        else:
            totalResult += output
    
    print(totalResult)

    return totalResult
