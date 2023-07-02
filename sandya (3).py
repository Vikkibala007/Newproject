# importing OpenCV(cv2) module
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter
 
# Save image in set directory
# Read RGB image
img = cv2.imread('coral.jpg')
 
# Output img with window name as 'image'
cv2.imshow('image', img)
 
# METHOD 1: RGB

# convert img to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# compute gamma = log(mid*255)/log(mean)
mid = 0.5
mean = np.mean(gray)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction
img_gamma1 = np.power(img, gamma).clip(0,255).astype(np.uint8)
# METHOD 2: HSV (or other color spaces)

# convert img to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue, sat, val = cv2.split(hsv)

# compute gamma = log(mid*255)/log(mean)
mid = 0.5
mean = np.mean(val)
gamma = math.log(mid*255)/math.log(mean)
print(gamma)

# do gamma correction on value channel
val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

# combine new value channel with original hue and sat channels
hsv_gamma = cv2.merge([hue, sat, val_gamma])
img_gamma2 = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

# show results
cv2.imshow('input', img)
cv2.imshow('result1', img_gamma1)
cv2.imshow('result2', img_gamma2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save results
cv2.imwrite('lioncuddle1_gamma1.jpg', img_gamma1)
cv2.imwrite('lioncuddle1_gamma2.jpg', img_gamma2)


  
# Cropping the image 
smol_image = image.crop((0, 0, 150, 150))
  
# Blurring on the cropped image
blurred_image = smol_image.filter(ImageFilter.GaussianBlur)
  
# Pasting the blurred image on the original image
image.paste(blurred_image, (0,0))
  
# Displaying the image
image.save('output.png')
