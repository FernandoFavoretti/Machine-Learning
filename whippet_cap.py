from __future__ import absolute_import, division, print_function, unicode_literals

from perceptron import MultiLayerPerceptron
from image_processor import ImageProcessor
from PIL import Image
import pi3d
import numpy as np
import random
import os

mlp = MultiLayerPerceptron(2048, 64, 5, 
                     wi_file='wi_file.npy', wo_file='wo_file.npy')
ip = ImageProcessor()

flist = []
for dname, dnames, fnames in os.walk('./pictures'):
  for f in fnames:
    if 'JPG' in f.upper():
      flist.append(os.path.join(dname, f))

def false_colour(a):
  X = [ 0.0,  0.25, 0.5,  0.75, 0.99, 1.0]
  R = [ 0.0,  0.0,255.0,255.0,255.0,255.0]
  G = [ 0.0,  0.0, 32.0,150.0,200.0,255.0]
  B = [64.0,120.0, 32.0, 32.0, 32.0,  0.0]
  #### 64.0 120.0 319.0 437.0 487.0 510.0
  new_a = np.zeros(a.shape[:2] + (3,), dtype=np.float)
  new_a[:,:,0] = np.interp(a, X, R)
  new_a[:,:,1] = np.interp(a, X, G)
  new_a[:,:,2] = np.interp(a, X, B)
  return new_a.astype(np.uint8)

grey = np.zeros((256, 256, 3), dtype=np.uint8)
sobel = np.zeros((256, 256, 3), dtype=np.uint8)
hog = np.zeros((16, 128, 3), dtype=np.uint8)
hidden = np.zeros((8, 8, 3), dtype=np.uint8)

DISPLAY = pi3d.Display.create(background=(0.1, 0.1, 0.1, 1.0))
CAM = pi3d.Camera()
shader = pi3d.Shader('uv_flat')
lgtshd = pi3d.Shader('mat_light')
pi3d.opengles.glDisable(pi3d.GL_CULL_FACE)

grey_tex = pi3d.Texture(grey, mipmap=False)
sobel_tex = pi3d.Texture(sobel, mipmap=False)
hog_tex = pi3d.Texture(hog, mipmap=False)
hidden_tex = pi3d.Texture(hidden)

grey_sprite = pi3d.ImageSprite(grey_tex, shader, w=2.5, h=2.5, x=-1.2, y=1.0, z=10.0)
sobel_sprite = pi3d.ImageSprite(sobel_tex, shader, w=2.5, h=2.5, x=-0.0, y=0.5, z=8.0)
hog_sprite = pi3d.ImageSprite(hog_tex, shader, w=2.5, h=2.5, x=1.0, y=0.0, z=6.0)
hidden_sprite = pi3d.ImageSprite(hidden_tex, shader, w=2.0, h=2.0, x=1.4, y=-0.5, z=4.0)

sobel_sprite.set_alpha(0.85)
hog_sprite.set_alpha(0.85)
hidden_sprite.set_alpha(0.85)

font = pi3d.Font('fonts/FreeSans.ttf', (150, 150, 150, 255))
str_list = [pi3d.String(font=font, string='Border Terrier', x=-1.0, y=1.0, z=3.0),
            pi3d.String(font=font, string='Flat Cap', x=-1.0, y=0.5, z=3.0),
            pi3d.String(font=font, string='Labrador', x=-1.0, y=0.0, z=3.0),
            pi3d.String(font=font, string='Top Hat', x=-1.0, y=-0.5, z=3.0),
            pi3d.String(font=font, string='Whippet', x=-1.0, y=-1.0, z=3.0)]
for st in str_list:
  st.set_shader(shader)

arrow = pi3d.Cone(radius=0.05, height=0.3, x=0.0, y=-2.0, z=3.0, rz=90.0)
arrow.set_shader(lgtshd)
arrow.set_material((1.0, 0.9, 0.4))

mykeys = pi3d.Keyboard()
mouse = pi3d.Mouse(restrict=False)
mouse.start()

fr = 1
while DISPLAY.loop_running():
  mx, my = mouse.position()
  mx *= -0.1
  my *= 0.1
  CAM.relocate(mx, my, [-0.5, 0.0, 5.0], [-6.0, -6.0, -6.0])

  grey_sprite.draw()
  sobel_sprite.draw()
  hog_sprite.draw()
  hidden_sprite.draw()
  for st in str_list:
    st.draw()
  arrow.draw()

  k = mykeys.read()
  if k >-1:
    if k == 27:
      mykeys.close()
      DISPLAY.destroy()
      break
    elif k == ord(' '):
      im = np.array(Image.open(random.choice(flist)))
      outlayer = mlp.feed_forward(ip.get_hog(im))
      grey_tex.update_ndarray(false_colour(ip.grey_scale))
      sobel_tex.update_ndarray(false_colour(ip.edges))
      hog_tex.update_ndarray(false_colour(ip.hog.reshape(16, 128)))
      hidden_tex.update_ndarray(false_colour(mlp.ah.reshape(8, 8)))
      arrow.positionY(1.0 - 0.5 * outlayer.argmax())
      str_colours = false_colour(outlayer.reshape(5,1)) / 255.0
      for i, st in enumerate(str_list):
        st.set_material(str_colours[i, 0])
    elif k == ord('p'):
      pi3d.screenshot("/home/patrick/Downloads/scr_caps_pi3d/scr_caps/fr{:03d}.jpg".format(fr))
      fr += 1

