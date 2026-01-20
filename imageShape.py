
import cv2

img = cv2.imread("input/input2.jpg")
if img is None:
    print("Error: Could not load image. Check the path.")
else:
    # Get the image's shape (height, width, channels)
    dimensions = img.shape

    # Extract height, width, and channels
    height = dimensions[0]
    width = dimensions[1]
    
    # For color images, there will be a third element for channels
    # For grayscale images, dimensions will only have height and width
    if len(dimensions) == 3:
        channels = dimensions[2]
        print(f"Image Dimension: {dimensions}")
        print(f"Image Height: {height}")
        print(f"Image Width: {width}")
        print(f"Number of Channels: {channels}")
    else:
        print(f"Image Dimension: {dimensions}")
        print(f"Image Height: {height}")
        print(f"Image Width: {width}")
        print("This is a grayscale image (no channel information).")