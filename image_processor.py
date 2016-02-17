from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy.ndimage
import random

class ImageProcessor(object):
  def __init__(self):
    self.grey_scale = np.zeros((256, 256), dtype=np.float)
    self.edges = np.zeros((256, 256), dtype=np.float)
    self.hog = np.zeros((2048,), dtype=np.float) # really 16,16,8 but flattened for feeding to NN
    ''' the two constants below are for mapping the pixel gradients in the
    two (256, 256) arrays (one for y direction, one for x direction) to
    the coordinates of the hog (histogram of gradients) array - see below
    in _make_hog()
    '''
    self.YNDX = np.array([[int(i/16)] * 256 for i in range(256)])
    self.XNDX = (self.YNDX).T

  def _process_image(self, im):
    ''' takes a 3D numpy array representation of the image as input. Then
    scales to 256,256 converts to grey scale and does edge detection.
    The images are held as float arrays from 0.0 to 1.0, for representation
    in the demo they are converted to 0-255 uint8 using a colourmap.
    '''
    if im.shape[:2] != (256, 256): # this is expecting 3D i.e. error for 'L' type image
      im = scipy.ndimage.interpolation.zoom(im, (256.0 / im.shape[0], 
                                            256.0 / im.shape[1], 1.0))
    im = im.astype(np.float)
    if len(im.shape) > 2: # convert to luminance - greyscale
      im = (im[:,:,:3] * [0.2989, 0.5870, 0.1140]).sum(axis=2)
    im /= 255.0
    self.grey_scale = im # keep a copy
    self.edges = scipy.ndimage.filters.sobel(im) # also generate edges file

  def _make_hog(self, im):
    hog = np.zeros((16, 16, 8))
    gr = np.gradient(im) # a list of two arrays, one for each axis
    # create an array of integers representing the direction of this gradient
    # this scales the angle to the range 0 to 7 as int
    bn = ((np.arctan2(gr[0], gr[1]) + np.pi) * 1.2732395).astype(np.int)
    # then accumulate the square of the gradient length in each bin
    hog[self.YNDX[:,:], self.XNDX[:,:], bn[:,:]] += gr[0][:,:] ** 2 + gr[1][:,:] ** 2
    # and finally square root the total
    hog = hog ** 0.5
    # then normalize by dividing by the max value
    hog /= hog.max()
    return hog.reshape(2048,)

  def get_hog(self, im):
    self._process_image(im)
    self.hog = self._make_hog(self.edges)
    return self.hog
    
  def get_hogs(self, img, number):
    ''' this method returns a generator for a number of hogs, each will
    be a variant of the img
    '''
    self._process_image(img)
    for j in range(number):
      # rotate
      a = scipy.ndimage.interpolation.rotate(self.edges, random.randint(0, 40) - 20)
      # scale
      a = scipy.ndimage.interpolation.zoom(a, (random.uniform(1.0, 1.075), 
                                               random.uniform(1.0, 1.075)))
      # flip
      a = a[:256,:256]
      # shift
      dx = random.randint(-20,20)
      dy = random.randint(-20,20)
      a = np.roll(a, dx, axis=1)
      a = np.roll(a, dy, axis=0)
      if random.randint(0, 1) == 1:
        a = a[:,::-1]
      yield self._make_hog(a)


