import cv2
from matplotlib import pyplot as plt

# Load images for stitching
image1 = cv2.imread('cabinet_pictures\cabinet1.PNG')
image2 = cv2.imread('cabinet_pictures\cabinet2.PNG')
image3 = cv2.imread('cabinet_pictures\cabinet3.PNG')

fig, ax = plt.subplots(1,2, figsize=(14,10))
ax[0].imshow(image1)
ax[0].set_title("part-1")
ax[0].axis("off")
ax[1].imshow(image2)
ax[1].set_title("part-2") 
ax[1].axis("off")
plt.show()

# Create a Stitcher object
stitcher = cv2.Stitcher_create()

# Stitch images
status, stitched_image = stitcher.stitch((image1, image2))

if status == cv2.Stitcher_OK:
    # Display the stitched image
    plt.figure(figsize = (14, 10))
    plt.imshow(stitched_image)
    plt.title('Stitched image')
    plt.axis('off')
    plt.show()
elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
    print('Not enough images for stitching.')
elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
    print('Homography estimation failed.')
else:
    print('Image stitching failed!')