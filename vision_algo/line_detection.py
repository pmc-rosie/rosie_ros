import enum
import statistics
import cv2
import numpy as np
import math


AREA_THREASHOLD = 0.5 # Percent

# ==============================================================================
# Init
# ==============================================================================
img = cv2.imread('rosie_ros/vision_algo/pic/Screenshot from 2024-06-19 21-52-26.png', 1)

# ==============================================================================
# Contour detection
# ==============================================================================
contrast = 3.0
brightness = 0.1
out = cv2.addWeighted( img, contrast, img, 1, brightness)
dst = cv2.Canny(out, 50, 200, None, 3)
ret,thresh = cv2.threshold(dst,0,250,0)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.08 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Filter for line-like shapes
    if len(approx) == 2:
        x1, y1 = approx[0][0]
        x2, y2 = approx[1][0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        ratio = length / (abs(y2 - y1) + 1e-5)  # Avoid division by zero

        if 3.5 < ratio < 10:
            filtered_lines.append(approx)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
# Draw all detected contours on the image
# cv2.drawContours(img, filtered_contours, -1, (255, 0, 0), 2)

# ==============================================================================
# False positive removal
# ==============================================================================
# # Find center
# rectangles = []
# for contour in filtered_contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     rectangles.append(((round(x+w/2), round(y+h/2)), contour))

# # Remove duplicates
# to_remove = []
# for i, rectangle in enumerate(rectangles):
#     center = rectangle[0]
#     for j in range(i+1, len(rectangles)):
#         x, y, w, h = cv2.boundingRect(rectangles[j][1])
#         if (center[0] > x and center[0] < x+w) and (center[1] > y and center[1] < h+y):
#             # print(f"Duplicate! | Index: {i} and {j} | Center position of {i} is: x:{center[0]}, y:{center[1]} which is inside range of {j} x:[{x};{x+w}] y:[{y};{y+h}]")
#             to_remove.append(i)
# for index in reversed(to_remove): # this way index don't change when removing elements
#     rectangles.pop(index)
     
# to_remove = []
# # Average color to keep only "white" lines
# for i, rectangle in enumerate(rectangles):
#     center = rectangle[0]
#     center_color = out[center[1]][center[0]].copy()
#     if not (center_color[0] >= 240 and center_color[1] >= 240 and center_color[2] >= 240):
#         to_remove.append(i)
# for index in reversed(to_remove): # this way index don't change when removing elements
#     rectangles.pop(index)
          
# # Find median value of area and only keep value close to median (It's estimated that as of now we mostly found the correct lines)
# avgs: list[float] = []
# for i, rectangle in enumerate(rectangles):
#     x, y, w, h = cv2.boundingRect(rectangle[1])
#     avgs.append(w*h)
# mean_area: float = statistics.median(avgs)
    
# to_remove = []
# for i, rectangle in enumerate(rectangles):
#     x, y, w, h = cv2.boundingRect(rectangle[1])
#     if (w*h) > mean_area + (mean_area * AREA_THREASHOLD) or (w*h) < mean_area - (mean_area * AREA_THREASHOLD):
#         print(f"#{i} of area: {w*h} isn't in range of {mean_area:.2f} ([{(mean_area - (mean_area * AREA_THREASHOLD)):.2f}->{(mean_area + (mean_area * AREA_THREASHOLD)):.2f}]")
#         to_remove.append(i)
# for index in reversed(to_remove): # this way index don't change when removing elements
#     rectangles.pop(index)
        
# ==============================================================================
# Display
# ============================================================================== 
# for index, rectangle in enumerate(rectangles):
#     x, y, w, h = cv2.boundingRect(rectangle[1])
#     center = rectangle[0]
    
#     cv2.circle(img, center, radius=1, color=(0, 255, 0), thickness=-1)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.putText(img, f'#:{index}, w: {w}, h: {h}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv2.putText(img, f'''#:{index} avg_color:{mean_color[index]})''', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv2.putText(img, f'''#:{index} avg_color:({round(mean_color[index][0])}, {round(mean_color[index][1])}, {round(mean_color[index][2])})''', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 1000, 1000)
cv2.imshow("win", img)
cv2.waitKey(0)

# cv2.namedWindow("win", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("win", 1000, 1000)
# cv2.imshow("win", gray)

cv2.waitKey()
