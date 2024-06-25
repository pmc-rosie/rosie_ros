import cv2
import numpy as np

img = cv2.imread('rosie_ros/vision_algo/pic/Screenshot from 2024-06-19 21-52-40.png', 1)

# define the contrast and brightness value
contrast = 3. # Contrast control ( 0 to 127)
brightness = 0.1 # Brightness control (0-100)

# call addWeighted function. use beta = 0 to effectively only operate on one image
out = cv2.addWeighted( img, contrast, img, 1, brightness)

# Stacking the original image with the enhanced image
result = np.hstack((img, out))
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 1000, 1000)
cv2.imshow('Result', result)
cv2.waitKey(0)
