import cv2
import numpy as np

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
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped
    else:
        return image

def stitch_images(image_files):
    # List to store aligned and cropped images
    aligned_images = []

    # Align and crop each image
    for image_file in image_files:
        image = cv2.imread(image_file)
        aligned_image = align_and_crop(image)
        aligned_images.append(aligned_image)

    # Determine the final output size
    max_width = max(image.shape[1] for image in aligned_images)
    total_height = sum(image.shape[0] for image in aligned_images)

    # Create the output image
    output_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    # Stitch the images together
    current_y = 0
    for image in aligned_images:
        height, width = image.shape[:2]
        output_image[current_y:current_y + height, :width] = image
        current_y += height

    return output_image

# List of image file paths to be stitched
image_files = ['cabinet_picture/cabinet1.PNG', 'cabinet_picture/cabinet2.PNG', 'cabinet_picture/cabinet3.PNG']

output_img = stitch_images(image_files)

# Save the final stitched image
final_image_path = 'aligned_stitched_image.png'
cv2.imwrite(final_image_path, output_img)

cv2.imshow('Stitched Image', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Aligned stitched image saved at {final_image_path}")
