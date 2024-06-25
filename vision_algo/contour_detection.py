import cv2
import numpy as np
import math

img = cv2.imread('rosie_ros/vision_algo/pic/Screenshot from 2024-06-19 21-52-40.png', 1)

# define the contrast and brightness value
contrast = 3. # Contrast control ( 0 to 127)
brightness = 0.1 # Brightness control (0-100)

# call addWeighted function. use beta = 0 to effectively only operate on one image
out = cv2.addWeighted( img, contrast, img, 1, brightness)

# Stacking the original image with the enhanced image
# result = np.hstack((img, out))
# cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Result', 1000, 1000)
# cv2.imshow('Result', result)
# cv2.waitKey(0)

## [edge_detection]
# Edge detection
dst = cv2.Canny(out, 50, 200, None, 3)
## [edge_detection]

cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 1000, 1000)
cv2.imshow("win", dst)

ret,thresh = cv2.threshold(dst,50,255,0)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.namedWindow("win", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("win", 1000, 1000)
# cv2.imshow("win", gray)

cv2.waitKey()
