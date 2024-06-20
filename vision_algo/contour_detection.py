import cv2
import numpy as np
import cv2
import numpy as np
import cv2
import numpy as np

# Load the image
image = cv2.imread('rosie_ros/vision_algo/pic/Screenshot from 2024-06-19 21-52-59.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Perform Hough line detection
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)

# Draw the detected lines on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
image = cv2.resize(image, (800, 600))
cv2.imshow('Horizontal Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
