from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import random

import skimage.feature
import skimage.filters
import skimage.transform

class ImageProcessor(object):
  def __init__(self):
    self.grey_scale = np.zeros((256, 256), dtype=np.uint8)
    self.edges = np.zeros((256, 256), dtype=np.uint8)
    self.hog = np.zeros((2048,))

  def _process_image(self, img):
    if img.shape[:2] != (256, 256):
      img = skimage.transform.resize(img, (256, 256))
    if len(img.shape) > 2:
      img = skimage.color.rgb2grey(img)
    self.grey_scale = img
    self.edges = skimage.filters.sobel(self.grey_scale)


  def get_hog(self, img):
    self._process_image(img)
    self.hog = skimage.feature.hog(self.edges, visualise=False, orientations=8, 
                                            pixels_per_cell=(16, 16), 
                                            cells_per_block=(1, 1))
    return self.hog
    
  def get_hogs(self, img, number):
    self._process_image(img)
    for j in range(number):
      a = skimage.transform.rotate(self.edges, random.randint(0, 40) - 20)
      a = skimage.transform.resize(a, (random.randint(256,275), random.randint(256,275)))
      a = a[:256,:256]
      if random.randint(0, 1) == 1:
        a = a[:,::-1]
      yield skimage.feature.hog(a, visualise=False, orientations=8, 
                                          pixels_per_cell=(16, 16), 
                                          cells_per_block=(1, 1))

