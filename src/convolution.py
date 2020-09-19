import numpy as np

class Convolution:
  def __init__(self, image = None, paddingSize = 2, filterCount = 1, filterSizeH = 3, filterSizeW = 3, strideSize = 3, filters = None):
    self.image = image
    self.paddingSize = paddingSize
    self.filterCount = filterCount
    self.filterSizeH = filterSizeH
    self.filterSizeW = filterSizeW
    self.strideSize = strideSize
    if filters is None:
        self.filters = np.random.randn(filterCount, filterSizeH, filterSizeW) / (filterSizeH * filterSizeW)
    else:
        self.filters = filters

  ### GETTER / SETTER ###
  def getImage(self):
      return self.image
  def setImage(self, image):
      self.image = image

  def getInputSize(self):
      return self.inputSize
  def setInputSize(self, inputSize):
      self.inputSize = inputSize

  def getPadding(self):
      return self.paddingSize
  def setPadding(self, paddingSize):
      self.paddingSize = paddingSize

  def getFilterCount(self):
      return self.filterCount
  def setFilterCount(self, filterCount):
      self.filterCount = filterCount

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
    result = np.zeros((self.image.shape[0] + (self.paddingSize * 2), self.image.shape[1] + (self.paddingSize * 2)))

    for i in range(self.paddingSize, (result.shape[0] - self.paddingSize)):
        for j in range(self.paddingSize, (result.shape[1] - self.paddingSize)):
            result[i, j] = self.image[i - self.paddingSize, j - self.paddingSize]

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
    # print('padding\n', padding)

    result = np.zeros(padding.shape)

    # print('filters :\n', self.filters)

    for curr_region, i, j in self.extract(padding):
        curr_result = np.tensordot(curr_region, self.filters)
        result[i, j] = np.sum(curr_result)
        #print(i, j, ':',"CURREGION : \n", curr_region, "FILTER :\n", self.filters, np.sum(curr_result))

    output = result[0:result.shape[0] - np.uint16(self.filterSizeH - 1):self.strideSize, 0:result.shape[1] - np.uint16(self.filterSizeW - 1):self.strideSize]

    return output
