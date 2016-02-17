from __future__ import absolute_import, division, print_function, unicode_literals

from MultiLayerPerceptron import MLP_Classifier
from image_processor import ImageProcessor
from PIL import Image
import pi3d
import numpy as np
import random

mlp = MLP_Classifier(2048, 64, 5, output_layer = 'logistic', 
                     wi_file='wi_file.npy', wo_file='wo_file.npy')
ip = ImageProcessor()
flist = ["./pictures/border/140561445231811882906801_BORDERTERRIER.jpg",
"./pictures/border/211fb1d06f9479a7650fc3bb47b93c8b_L.jpg",
"./pictures/border/bCM07nN-730x430.jpg",
"./pictures/border/border-terrier-1.jpg",
"./pictures/border/Border Terrier 9R011D-060.JPG",
"./pictures/border/border_terrier.jpg",
"./pictures/border/border-terrier.jpg",
"./pictures/border/border-terrier-running-116455_x.jpg",
"./pictures/border/borderterriersf6.jpg",
"./pictures/border/file_23058_border-terrier.jpg",
"./pictures/border/gilli_lucy_border_terriers_10_500.jpg",
"./pictures/border/images (1).jpg",
"./pictures/border/images (2).jpg",
"./pictures/border/images (3).jpg",
"./pictures/border/images (4).jpg",
"./pictures/border/images (5).jpg",
"./pictures/border/images (6).jpg",
"./pictures/border/images.jpg",
"./pictures/border/sketch101.jpg",
"./pictures/border/sketch103.jpg",
"./pictures/border/sketch104.jpg",
"./pictures/border/sketch105.jpg",
"./pictures/border/sketch106.jpg",
"./pictures/border/sketch107.jpg",
"./pictures/border/sketch108.jpg",
"./pictures/border/Sketch140215515.jpg",
"./pictures/border/Sketch140215751.jpg",
"./pictures/border/Sketch140221559.jpg",
"./pictures/border/Sketch14022343.jpg",
"./pictures/border/wellgr.jpg",

"./pictures/flat_cap/05c3c413c9e611a0b47b3a079366c552.jpg",
"./pictures/flat_cap/180007_13.jpg",
"./pictures/flat_cap/501.jpg",
"./pictures/flat_cap/51tpeqvYmCL.jpg",
"./pictures/flat_cap/51YwBY-fU7L._AC_UL200_SR160,200_.jpg",
"./pictures/flat_cap/683139274925911cd900f7b4ce61f307.jpg",
"./pictures/flat_cap/73582755.jpg",
"./pictures/flat_cap/acfa7c9b235960d00e83cf3a00bf663c.jpg",
"./pictures/flat_cap/Artimus-web-5.jpg",
"./pictures/flat_cap/Copy_of_Sketch140221645.jpg",
"./pictures/flat_cap/flat_cap_black.jpg",
"./pictures/flat_cap/flatcapherringboneblack.jpg",
"./pictures/flat_cap/grey-tape-stripe-flat-cap-debenhams-22.jpg",
"./pictures/flat_cap/images (1).jpg",
"./pictures/flat_cap/images (2).jpg",
"./pictures/flat_cap/images.jpg",
"./pictures/flat_cap/img-thing.jpg",
"./pictures/flat_cap/jb-stetson-black-patch-flat-cap-31.jpg",
"./pictures/flat_cap/sk01.jpg",
"./pictures/flat_cap/sk02.jpg",
"./pictures/flat_cap/sk05.jpg",
"./pictures/flat_cap/sk06.jpg",
"./pictures/flat_cap/sk07.jpg",
"./pictures/flat_cap/sk08.jpg",
"./pictures/flat_cap/sk09.jpg",
"./pictures/flat_cap/sk10.jpg",
"./pictures/flat_cap/sk11.jpg",
"./pictures/flat_cap/sk12.jpg",
"./pictures/flat_cap/sk13.jpg",
"./pictures/flat_cap/Sketch140215848.jpg",
"./pictures/flat_cap/Sketch14022180.jpg",
"./pictures/flat_cap/sko3.jpg",
"./pictures/flat_cap/STE6627301-61_3,stetson,stetson-brown-flat-cap.jpg",
"./pictures/flat_cap/stetson-brown-herringbone-hatteras-flat-cap-product-1-21798196-3-368332051-normal.jpeg",
"./pictures/flat_cap/stetson-muskagon-duckbill-cap-340-p.jpg",

"./pictures/labrador/05_labrador1.jpg",
"./pictures/labrador/05_labrador.jpg",
"./pictures/labrador/28268-004-1E90A448.jpg",
"./pictures/labrador/628245-labrador-retriever-wallpaper.jpg",
"./pictures/labrador/8105-004-2F41FD96.jpg",
"./pictures/labrador/all-about-the-labrador-retriever-5217858f18edd.jpg",
"./pictures/labrador/charakterystyka-labrador-1.jpg",
"./pictures/labrador/english-labrador-retriever-pictures-323.jpg",
"./pictures/labrador/file_22988_labrador-retriever-460x290.jpg",
"./pictures/labrador/images (1).jpg",
"./pictures/labrador/images (2).jpg",
"./pictures/labrador/images (4).jpg",
"./pictures/labrador/images (5).jpg",
"./pictures/labrador/images (6).jpg",
"./pictures/labrador/images.jpg",
"./pictures/labrador/labrador-1.jpg",
"./pictures/labrador/labrador-build.jpg",
"./pictures/labrador/labrador-kremowy-retrievera-2329118.jpg",
"./pictures/labrador/LabradorRetrieverBlackPurebredDogSonny8YearsOldTim1.JPG",
"./pictures/labrador/labrador-retriever-dog.jpg",
"./pictures/labrador/MZ00001H3-Labrador.jpg",
"./pictures/labrador/sketch109.jpg",
"./pictures/labrador/sketch110.jpg",
"./pictures/labrador/sketch111.jpg",
"./pictures/labrador/Sketch140221436.jpg",
"./pictures/labrador/vectra_labrador.jpg",
"./pictures/labrador/white-labrador-retriever.jpg",

"./pictures/top_hat/26367HeadnHomeStovePiper_2.jpg",
"./pictures/top_hat/26374HeadnHomeTopper_3.jpg",
"./pictures/top_hat/56177-large.jpg",
"./pictures/top_hat/8T6zK5jTE.jpg",
"./pictures/top_hat/download.jpg",
"./pictures/top_hat/drab_shell_top_hat_grey_2.jpg",
"./pictures/top_hat/great-and-powerful-oscar-diggs-top-hat-.jpg",
"./pictures/top_hat/hat_top_silk.jpg",
"./pictures/top_hat/heathen hat.png",
"./pictures/top_hat/i50c955a093c76ad249d64c0d76f585db.jpg",
"./pictures/top_hat/images (10).jpg",
"./pictures/top_hat/images (1).jpg",
"./pictures/top_hat/images (8).jpg",
"./pictures/top_hat/images (9).jpg",
"./pictures/top_hat/images.jpg",
"./pictures/top_hat/m341ouPE9eH8YYfmKjBRUCw.jpg",
"./pictures/top_hat/sk01.jpg",
"./pictures/top_hat/sk02.jpg",
"./pictures/top_hat/sk03.jpg",
"./pictures/top_hat/sk04.jpg",
"./pictures/top_hat/sk05.jpg",
"./pictures/top_hat/sk06.jpg",
"./pictures/top_hat/sk07.jpg",
"./pictures/top_hat/sk09.jpg",
"./pictures/top_hat/sk10.jpg",
"./pictures/top_hat/sk11.jpg",
"./pictures/top_hat/sk12.jpg",
"./pictures/top_hat/Sketch14022106.jpg",
"./pictures/top_hat/stock-illustration-24883000-top-hat-vector-illustration.jpg",
"./pictures/top_hat/th3.jpg",
"./pictures/top_hat/tophat2.jpg",
"./pictures/top_hat/TopHat2_jpga837fbd9-017a-4abe-910c-6479df186d1fLarger_zpsea26859f.jpg",
"./pictures/top_hat/top_hat_by_timgoransson-d4n6fwm.jpg",

"./pictures/whippet/1280px-Greyhound_Racing_2_amk.jpg",
"./pictures/whippet/1280px-Szombierki_greyhound_18.09.2011_2pl.jpg",
"./pictures/whippet/220px-WhippetWhiteSaddled_wb.jpg",
"./pictures/whippet/27177-Brindle-and-white-Whippet-pup-white-background.jpg",
"./pictures/whippet/4524881500.jpg",
"./pictures/whippet/blue-whippet-stud-dog-5207e281c19f6.JPG",
"./pictures/whippet/sk01.jpg",
"./pictures/whippet/sk02.jpg",
"./pictures/whippet/sk03.jpg",
"./pictures/whippet/sk04.jpg",
"./pictures/whippet/sk05.jpg",
"./pictures/whippet/sk06.jpg",
"./pictures/whippet/sk07.jpg",
"./pictures/whippet/sk09.jpg",
"./pictures/whippet/sk10.jpg",
"./pictures/whippet/Sketch140221114.jpg",
"./pictures/whippet/Sketch140222024.jpg",
"./pictures/whippet/Sketch140222232.jpg",
"./pictures/whippet/Sketch140223040.jpg",
"./pictures/whippet/Sketch14022542.jpg",
"./pictures/whippet/Sketch14022658.jpg",
"./pictures/whippet/Sketch14022934.jpg",
"./pictures/whippet/whippet_01_lg.jpg",
"./pictures/whippet/whippet1.jpg",
"./pictures/whippet/whippet-2.jpg",
"./pictures/whippet/whippet2.jpg",
"./pictures/whippet/whippet3.jpg",
"./pictures/whippet/whippet4.jpg",
"./pictures/whippet/whippet5.jpg",
"./pictures/whippet/whippet-in-the-wind-liane-weyers.jpg",
"./pictures/whippet/Whippet.jpg",
"./pictures/whippet/WhippetPiperPicture041compressed.JPG",
"./pictures/whippet/whippet-portrait-white-background-27755329.jpg"]
flist1 = [
"./pictures/border-terrier-u3.jpg",
"./pictures/images (3).jpg",
"./pictures/images (7).jpg",
"./pictures/linen-ivy-league-flat-cap-brown.jpg",
"./pictures/maxresdefault.jpg",
"./pictures/sk04.jpg",
"./pictures/sk08a.jpg",
"./pictures/sk08.jpg",
"./pictures/sketch102.jpg"
]
def false_colour(a):
  X = [0.0, 0.25, 0.5, 0.75, 1.0]
  R = [0.0, 10.0, 30.0, 86.0, 255.0]
  G = [0.0, 20.0, 80.0, 120.0, 30.0]
  B = [60.0, 50.0, 30.0, 4.0, 30.0]
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

grey_sprite = pi3d.ImageSprite(grey_tex, shader, w=2.5, h=2.5, x=-1.0, z=10.0)
sobel_sprite = pi3d.ImageSprite(sobel_tex, shader, w=2.5, h=2.5, x=-0.5, z=8.0)
hog_sprite = pi3d.ImageSprite(hog_tex, shader, w=2.5, h=2.5, x=0.5, z=6.0)
hidden_sprite = pi3d.ImageSprite(hidden_tex, shader, w=2.0, h=2.0, x=1.0, z=4.0)

sobel_sprite.set_alpha(0.85)
hog_sprite.set_alpha(0.85)
hidden_sprite.set_alpha(0.85)

font = pi3d.Font('fonts/FreeSans.ttf', (150, 150, 150, 255))
str_list = [pi3d.String(font=font, string='Border Terrier', x=1.0, y=1.0, z=3.0),
            pi3d.String(font=font, string='Flat Cap', x=1.0, y=0.5, z=3.0),
            pi3d.String(font=font, string='Labrador', x=1.0, y=0.0, z=3.0),
            pi3d.String(font=font, string='Top Hat', x=1.0, y=-0.5, z=3.0),
            pi3d.String(font=font, string='Whippet', x=1.0, y=-1.0, z=3.0)]
for st in str_list:
  st.set_shader(shader)

arrow = pi3d.Cone(radius=0.05, height=0.3, x=2.0, y=-2.0, z=3.0, rz=90.0)
arrow.set_shader(lgtshd)
arrow.set_material((1.0, 0.9, 0.4))

mykeys = pi3d.Keyboard()
mouse = pi3d.Mouse(restrict=False)
mouse.start()
alt = True
fr = 1
while DISPLAY.loop_running():
  mx, my = mouse.position()
  mx *= -0.1
  my *= 0.1
  CAM.relocate(mx, my, [0.0, 0.0, 5.0], [-6.0, -6.0, -6.0])

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
      outlayer = mlp.feedForward(ip.get_hog(im))
      grey_tex.update_ndarray(false_colour(ip.grey_scale))
      sobel_tex.update_ndarray(false_colour(ip.edges))
      hog_tex.update_ndarray(false_colour(ip.hog.reshape(16, 128)))
      hidden_tex.update_ndarray(false_colour(mlp.ah.reshape(8, 8)))
      arrow.positionY(1.0 - 0.5 * outlayer.argmax())
      str_colours = false_colour(outlayer.reshape(5,1)) / 255.0
      for i, st in enumerate(str_list):
        st.set_material(str_colours[i, 0])

  #pi3d.screenshot("/home/patrick/Downloads/scr_caps_pi3d/scr_caps/fr{:03d}.jpg".format(fr))
  #fr += 1
