from __future__ import absolute_import, division, print_function, unicode_literals

from PIL import Image
import os
import numpy as np

from image_processor import ImageProcessor

result_dict = {
    './pictures/border':'1,0,0,0,0,',
    './pictures/flat_cap':'0,1,0,0,0,',
    './pictures/labrador':'0,0,1,0,0,',
    './pictures/top_hat':'0,0,0,1,0,',
    './pictures/whippet':'0,0,0,0,1,'}
ip = ImageProcessor()
with open('pictures/learndb.csv','w') as db:
  for dname, dnames, fnames in os.walk('./pictures'):
    if dname != './pictures' and len(fnames) > 0 and not '__' in dname:
      for i,f in enumerate(fnames):
        hogs = ip.get_hogs(np.array(Image.open(os.path.join(dname, f))), 20)
        for hog in hogs:
          db.write(result_dict[dname])
          db.write(','.join(['0' if k < 0.00001 and k > -0.00001 else '{:5.5f}'.format(k) for k in hog]))
          db.write(chr(10))
          print(dname, i)
