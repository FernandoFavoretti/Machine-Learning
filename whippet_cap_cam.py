from __future__ import absolute_import, division, print_function, unicode_literals

from perceptron import MultiLayerPerceptron
from image_processor import ImageProcessor
from PIL import Image 
import pi3d
import numpy as np
import threading
import time
import picamera
import picamera.array

mlp = MultiLayerPerceptron(2048, 64, 5, output_layer = 'logistic', 
                     wi_file='wi_file.npy', wo_file='wo_file.npy')
ip = ImageProcessor()
ready_flag = False
outlayer = None

def get_image():
  global mlp, ip, outlayer, ready_flag
  with picamera.PiCamera() as camera:
    camera.resolution = (256, 256)
    # try to fix the exposure to stop the image 'changing'
    camera.frame_rate = 20
    camera.iso = 400
    time.sleep(1.0)
    camera.shutter_speed = camera.exposure_speed
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    with picamera.array.PiRGBArray(camera) as output:
      while True: # loop for ever
        if not ready_flag:
          output.truncate(0)
          camera.capture(output, 'rgb')
          im = output.array
          hog = ip.get_hog(im)
          outlayer = mlp.feed_forward(hog)
          ready_flag = True
        time.sleep(0.1) 

t = threading.Thread(target=get_image)
t.daemon = True
t.start()

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

########################################################################
DISPLAY = pi3d.Display.create(x=250, y=50, background=(0.1, 0.1, 0.1, 1.0))
CAM = pi3d.Camera()
shader = pi3d.Shader('uv_flat')
lgtshd = pi3d.Shader('mat_light')
pi3d.opengles.glDisable(pi3d.GL_CULL_FACE)

grey_tex = pi3d.Texture(grey, mipmap=False)
sobel_tex = pi3d.Texture(sobel, mipmap=False)
hog_tex = pi3d.Texture(hog, mipmap=False)
hidden_tex = pi3d.Texture(hidden, mipmap=True)

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

arrow_new = arrow_y = -2.0
arrow = pi3d.Cone(radius=0.05, height=0.3, x=0.0, y=arrow_y, z=3.0, rz=90.0)
arrow.set_shader(lgtshd)
arrow.set_material((1.0, 0.9, 0.4))

mykeys = pi3d.Keyboard()
mouse = pi3d.Mouse(restrict=False)
mouse.start()

fr = 0
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
  arrow_y = arrow_y * 0.995 + 0.005 * arrow_new
  arrow.positionY(arrow_y)
  arrow.draw()

  if ready_flag:
    # new image available, update textures
    grey_tex.update_ndarray(false_colour(ip.grey_scale))
    sobel_tex.update_ndarray(false_colour(ip.edges))
    hog_tex.update_ndarray(false_colour(ip.hog.reshape(16, 128)))
    hidden_tex.update_ndarray(false_colour(mlp.ah.reshape(8, 8)))
    # target position for arrow
    arrow_new = 1.0 - 0.5 * outlayer.argmax()
    # colour and size of possible answers
    str_colours = false_colour(outlayer.reshape(5,1)) / 255.0
    for i, st in enumerate(str_list):
      st.set_material(str_colours[i, 0])
      st.scale(0.5 + outlayer[i], 1.0, 1.0)
    ready_flag = False

  k = mykeys.read()
  if k >-1:
    if k == 27:
      mykeys.close()
      DISPLAY.destroy()
      break

  ''' NB uncomment below with care; this will generate a lot of images very 
  quickly! Also you obviously need a viable path rather than the one below 
  which will only match the USB drive I happen to be using.
  '''
  #pi3d.screenshot("/media/pi/701E-64FC/tmp/scrap/fr{:05d}.jpg".format(fr))
  #fr += 1
