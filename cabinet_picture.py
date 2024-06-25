import cv2
import numpy as np

# Load images
image1 = cv2.imread('cabinet_pictures\cabinet1.PNG')
image2 = cv2.imread('cabinet_pictures\cabinet2.PNG')
image3 = cv2.imread('cabinet_pictures\cabinet3.PNG')

def align_and_crop(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Approximate the contour to a quadrilateral
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        # Get the points in the correct order
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # Calculate the width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Define the destination points for perspective transform
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Apply the perspective transformation
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, 2*maxHeight))

        return warped
    else:
        return image

# Create a list of images
images = [image1, image2, image3]

# Initialize the stitcher
stitcher = cv2.Stitcher_create()

# Stitch the images
status, stitched = stitcher.stitch(images)

# Check if the stitching was successful
if status == cv2.Stitcher_OK:
    # Align and crop the stitched image
    final_image = align_and_crop(stitched)
    
    # Save the final image
    final_image_path = 'aligned_stitched_image.png'
    cv2.imwrite(final_image_path, final_image)
    print(f"Aligned stitched image saved at {final_image_path}")
else:
    print("Error during stitching process")
