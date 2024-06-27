import enum
import statistics
import cv2
import numpy as np
import math


AREA_THRESHOLD = 1.5 # Percent
LENGTH_THRESHOLD = 0.05 # Percent

# threshold in percent 20%
def IN_THRESHOLD(value_: float, target_: float, threshold_: float):
    error = value_ - abs(target_)
    error_percent = error/abs(target_) * 100
    if (error_percent < abs(threshold_)):
        return True
    else:
        return False

class Point:
    def __init__(self, x_, y_) -> None:
        self.x: int = x_
        self.y: int = y_

class Line:
    def __init__(self, x0_: int, y0_:int, x2_, y2_: int) -> None:
        self.origin: Point = Point(x0_, y0_)
        self.start: Point = self.origin
        self.end: Point = Point(x2_, y2_)
        self.center: Point = Point(round(x0_ + (x2_-x0_)/2), round(y0_ + (y2_-y0_)/2))
        self.length: int = round(np.sqrt((x2_ - x0_)**2 + (y2_ - y0_)**2))

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

filtered_contours = []
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        area = w*h
        ratio = w/h
        if ratio > 3.5 and ratio < 10:
            filtered_contours.append(approx)
            
# Draw all detected contours on the image
# cv2.drawContours(img, filtered_contours, -1, (255, 0, 0), 2)

# ==============================================================================
# False positive removal
# ==============================================================================
# Find center
rectangles = []
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    rectangles.append(((round(x+w/2), round(y+h/2)), contour))

# Remove duplicates
to_remove = []
for i, rectangle in enumerate(rectangles):
    center = rectangle[0]
    for j in range(i+1, len(rectangles)):
        x, y, w, h = cv2.boundingRect(rectangles[j][1])
        if (center[0] > x and center[0] < x+w) and (center[1] > y and center[1] < h+y):
            # print(f"Duplicate! | Index: {i} and {j} | Center position of {i} is: x:{center[0]}, y:{center[1]} which is inside range of {j} x:[{x};{x+w}] y:[{y};{y+h}]")
            to_remove.append(i)
for index in reversed(to_remove): # this way index don't change when removing elements
    rectangles.pop(index)
     
# Average color to keep only "white" lines
to_remove = []
for i, rectangle in enumerate(rectangles):
    center = rectangle[0]
    center_color = out[center[1]][center[0]].copy()
    if not (center_color[0] >= 255 and center_color[1] >= 255 and center_color[2] >= 255):
        to_remove.append(i)
for index in reversed(to_remove): # this way index don't change when removing elements
    rectangles.pop(index)
          
# Find median value of area and only keep value close to median (It's estimated that as of now we mostly found the correct lines)
avgs: list[float] = []
for i, rectangle in enumerate(rectangles):
    x, y, w, h = cv2.boundingRect(rectangle[1])
    avgs.append(w*h)
mean_area: float = statistics.median(avgs)
    
to_remove = []
for i, rectangle in enumerate(rectangles):
    x, y, w, h = cv2.boundingRect(rectangle[1])
    if (w*h) > mean_area + (mean_area * AREA_THRESHOLD) or (w*h) < mean_area - (mean_area * AREA_THRESHOLD):
        print(f"#{i} of area: {w*h} isn't in range of {mean_area:.2f} ([{(mean_area - (mean_area * AREA_THRESHOLD)):.2f}->{(mean_area + (mean_area * AREA_THRESHOLD)):.2f}]")
        to_remove.append(i)
for index in reversed(to_remove): # this way index don't change when removing elements
    rectangles.pop(index)
    
# ==============================================================================
# Finding lines matching the found rectangles
# ==============================================================================
lines: list[Line] = []
for contour in contours:
    epsilon = 0.08 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 2:
        x0, y0 = approx[0][0]
        x1, y1 = approx[1][0]
        line = Line(x0, y0, x1, y1)
        lines.append(line)
        
# Only keep horizontal (mostly) lines
to_remove: list[int] = []
for i, line in enumerate(lines):
    if (abs(line.end.x - line.start.x)) < (abs(line.end.y - line.start.y)):
        to_remove.append(i)
        # print(f"Removing line: width: {abs(line.end.x - line.start.x)} is < than height: {abs(line.end.y - line.start.y)}")
for index in reversed(to_remove):
    lines.pop(index)
    
# for line in lines:
    # cv2.line(img, (line.start.x, line.start.y), (line.end.x, line.end.y), (255, 0, 0))
        
# Detect center color to keep only "white" lines
to_remove = []
for i, line in enumerate(lines):
    center_color = out[line.center.y][line.center.x].copy()
    if not (center_color[0] >= 255 and center_color[1] >= 255 and center_color[2] >= 255):
        to_remove.append(i)
for index in reversed(to_remove): # this way index don't change when removing elements
    lines.pop(index)
    
# Find average width of previously detected rectangles and keep only line corresponding
avg_width: float = 0
for rectangle in rectangles:
    x, y, w, h = cv2.boundingRect(rectangle[1])
    avg_width += w
avg_width /= len(rectangles)

for line in lines:
    if IN_THRESHOLD(float(line.length), avg_width, LENGTH_THRESHOLD):
        cv2.line(img, (line.start.x, line.start.y), (line.end.x, line.end.y), (255, 0, 0))
        
# ==============================================================================
# Display
# ============================================================================== 
for index, rectangle in enumerate(rectangles):
    x, y, w, h = cv2.boundingRect(rectangle[1])
    center = rectangle[0]
    
    cv2.circle(img, center, radius=1, color=(0, 255, 0), thickness=-1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText(img, f'#:{index}, w: {w}, h: {h}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # cv2.putText(img, f'''#:{index} avg_color:{mean_color[index]})''', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # cv2.putText(img, f'''#:{index} avg_color:({round(mean_color[index][0])}, {round(mean_color[index][1])}, {round(mean_color[index][2])})''', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.namedWindow("win", cv2.WINDOW_NORMAL)
cv2.resizeWindow("win", 1000, 1000)
cv2.imshow("win", img)
cv2.waitKey(0)

# cv2.namedWindow("win", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("win", 1000, 1000)
# cv2.imshow("win", gray)

cv2.waitKey()
