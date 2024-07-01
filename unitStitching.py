import cv2
import numpy as np
import os

# Function used to create multiple images from one image, used for tests
# The output of the function is what the robot would input (multiple individual image files, each associated to a unit)
def crop_tests(image_multiple_units, crop_size=(1100, 120), num_crops=10,  output_dir='cropped_images'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    cropped_units_tests = []
    image = cv2.imread(image_multiple_units[0])

    # Get base filename without extension
    filename = os.path.splitext(os.path.basename(image_multiple_units[0]))[0]

    for units in range(num_crops):
        top_left_x = 0
        top_left_y = (units * crop_size[1]) - (units*30)
        cropped_image = image[top_left_y:top_left_y + crop_size[1], top_left_x:top_left_x + crop_size[0]]
        
        # Generate output filename
        output_filename = f"{filename}_crop{units+1}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)

        # Append cropped image to list
        cropped_units_tests.append(cropped_image)

    return cropped_units_tests

# Function to crop the image of the unit to the desired size (a little over crop to compensate)
def overcrop(input_unit, crop_size=(1000, 120)):
    unit = cv2.imread(input_unit)
    top_left_x = 100
    top_left_y = 0
    overcropped_unit = unit[top_left_y:top_left_y + crop_size[1], top_left_x:top_left_x + crop_size[0]]
    cv2.imwrite(input_unit, overcropped_unit)

    return overcropped_unit

# Function to detecte borders in image if possible
def crop_to_border(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour -- PROBLEME LORSQUE UNITÉ VIDE, PCQ PENSE QUE LE RECTANGLE EST UN CONTOUR
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
        top_left_x = 0
        top_left_y = 5
        overcropped_unit = image[top_left_y:top_left_y + 90, top_left_x:top_left_x + 1000]

        return overcropped_unit
    
# Function to stitch the cropped images together
def stitch_images(image_files):

    # Determine the final output size
    max_width = 1000
    total_height = 100*10

    # Create the output image
    output_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    # Stitch the images together
    current_y = 0
    for unit in image_files:
        height, width = unit.shape[:2]
        output_image[current_y:current_y + height, :width] = unit
        current_y += height

    return output_image

# Main
# Split image into units, step necessary only in tests to recreate what we would normally have as input
image_multiple_units = ['cabinet_pictures/cabinet1.PNG']
input_units = crop_tests(image_multiple_units)
input_units = ['cropped_images/cabinet1_crop1.jpg', 'cropped_images/cabinet1_crop2.jpg', 'cropped_images/cabinet1_crop3.jpg', 'cropped_images/cabinet1_crop4.jpg', 'cropped_images/cabinet1_crop5.jpg', 'cropped_images/cabinet1_crop6.jpg', 'cropped_images/cabinet1_crop7.jpg', 'cropped_images/cabinet1_crop8.jpg', 'cropped_images/cabinet1_crop9.jpg', 'cropped_images/cabinet1_crop10.jpg']
overcropped_units = []
cropped_units = []

# Overcrop each input unit, then detect borders if possible
for units in range(len(input_units)):
    overcropped_units.append(overcrop(input_units[units]))
    cropped_units.append(crop_to_border(overcropped_units[units])) #gérer le dernier crop

output_img = stitch_images(cropped_units)

# Save the final stitched image
final_image_path = 'aligned_stitched_image.png'
cv2.imwrite(final_image_path, output_img)

cv2.imshow('Stitched Image', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Aligned stitched image saved at {final_image_path}")
