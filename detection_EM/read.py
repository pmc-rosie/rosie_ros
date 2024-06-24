import cv2 as cv
import numpy as np
import os
from pyzbar import pyzbar
import shutil


image_name = "image3"
UPSCALE_FACTOR = 2

# Name of the outpu directory
output_dir = "detection_EM/output_tracking_numbers"

# Remove the existing output_tracking_numbers directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# Create a directory to save the cropped images
os.makedirs(output_dir)

# Load the image
img_path = "detection_EM/images/" + image_name + ".jpg"
img = cv.imread(img_path)

print(img_path)

# Check if the image is loaded successfully
if img is None:
    raise Exception("Image not loaded.")

# Print the image currently being scanned
print("\nScanning: " + img_path.split('/')[1] + "\n")

# Create a copy of the original image to use for ROI extraction
img_copy = img.copy()

# Convert the image to HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define the range for yellow color in HSV
lower_yellow = np.array([20, 125, 150])
upper_yellow = np.array([40, 255, 255])

# Create a mask for yellow color
mask = cv.inRange(hsv, lower_yellow, upper_yellow)

# Find contours in the mask
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

success = 0
fail = 0
index = 0

# Loop over the contours to draw bounding boxes and scan barcodes
for contour in contours:
    # Get the bounding box for each contour
    x, y, w, h = cv.boundingRect(contour)
    
    # Check if the contour is at least 40 pixels by 10 pixels
    if w >= 40 and h >= 10:
        # Draw the bounding box on the original image
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the region of interest (ROI) from the copy of the original image
        roi = img_copy[y:y+h, x:x+w]

        # Convert ROI to grayscale
        gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        # Upscale the ROI using bilinear interpolation
        upscale_factor = UPSCALE_FACTOR  # You can change this factor as needed
        roi_upscaled = cv.resize(gray_roi, None, fx=upscale_factor, fy=upscale_factor, interpolation=cv.INTER_LINEAR)       

        # Save the upscaled ROI as a separate image
        roi_filename = os.path.join(output_dir, f"Tracking number {index + 1}.jpg")

        # Scan the ROI for barcodes
        barcodes = pyzbar.decode(roi_upscaled)
        if barcodes:
            for barcode in barcodes:
                barcode_data = barcode.data.decode("utf-8")
                print(f"Tracking number {index+1}: \033[1;32m{barcode_data}\033[0m")

                # Draw barcode data on the image
                barcode_text = f"{barcode_data}"
                cv.putText(img, barcode_text, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # If barcode detected, save with barcode name
                roi_filename = os.path.join(output_dir, f"{barcode_data}.jpg")

                success += 1
        else:
            cv.putText(img, f"Tracking number {index + 1}", (x-100, y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Tracking number {index+1}: \033[1;31m--------\033[0m")

            fail += 1
        
        cv.imwrite(roi_filename, roi_upscaled)

        index += 1

# Calculate and print the success rate
if fail == 0:
    print("\nSuccess rate: \033[1;32m100%\033[0m\n")
elif success == 0:
    print("\nSuccess rate: \033[1;31m0%\033[0m\n")
else:
    success_rate = "{:.1f}".format(float(success) / float(success + fail) * 100)
    print(f"\nSuccess rate: \033[1;33m{success_rate}%\033[0m\n")

# Save the original image with bounding boxes and text
annotated_img_filename = os.path.join(output_dir, "__Scanned image.jpg")
cv.imwrite(annotated_img_filename, img)

# Display the original image with bounding boxes and text
# cv.imshow("Yellow Squares", img)

# Wait for a key press indefinitely
# cv.waitKey(5000)
cv.destroyAllWindows()
