import numpy as np

class Convolution:
  def __init__(self, image = None, inputSize = 1, paddingSize = 0, filterCount = 1, filterSize = 3, strideSize = 1):
    self.image = image
    self.inputSize = inputSize
    self.paddingSize = paddingSize
    self.filterCount = filterCount
    self.filterSize = filterSize
    self.strideSize = strideSize
    self.filters = np.random.randn(filterCount, filterSize, filterSize) / (filterSize ^ 2)

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
      return self.filterSize
  def setFilterSize(self, filterSize):
      self.filterSize = filterSize

  def getStride(self):
      return self.strideSize
  def setStride(self, strideSize):
      self.strideSize = strideSize

  def extract(self):
    '''
    Generates all possible i x j image regions using padding.
    - image is a 2d numpy array.
    '''
    h, w = self.image.shape

    for i in range(h - (self.inputSize - self.paddingSize)):
      for j in range(w - (self.inputSize - self.paddingSize)):
        region = self.image[i:(i + self.inputSize + self.paddingSize), j:(j + self.inputSize + self.paddingSize)]

        yield region, i, j

  def forward(self):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, filterCount).
    '''
    result = np.zeros((self.image.shape))

    for curr_region, i, j in self.extract():
      curr_result = curr_region * self.filters
      result[i, j] = np.sum(curr_result)

    output = result[np.uint16(self.filterSize / 2):result.shape[0] - np.uint16(self.filterSize / 2), np.uint16(self.filterSize / 2):result.shape[1] - np.uint16(self.filterSize / 2)]

    return output
