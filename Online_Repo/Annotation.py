# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:35:23 2024

@author: UOU
"""

import cv2
import numpy as np

# Parameters for drawing
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial x, y coordinates of the region

# List to store segmentation points
annotations = []

# Mouse callback function to draw contours
def draw_contour(event, x, y, flags, param):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        annotations.append([(x, y)])  # Start a new contour

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Add points to the current contour
            annotations[-1].append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Close the contour by connecting the last point to the first
        annotations[-1].append((x, y))

# Function to display the image and collect annotations
def segment_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return

    # Create a clone of the image for annotation display
    annotated_image = image.copy()
    cv2.namedWindow("Image Segmentation")
    cv2.setMouseCallback("Image Segmentation", draw_contour)

    while True:
        # Show the annotations on the cloned image
        temp_image = annotated_image.copy()
        for contour in annotations:
            points = np.array(contour, dtype=np.int32)
            cv2.polylines(temp_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the image with annotations
        cv2.imshow("Image Segmentation", temp_image)
        
        # Press 's' to save annotations, 'c' to clear, and 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Save annotations
            with open("annotations.txt", "w") as f:
                for contour in annotations:
                    f.write(str(contour) + "\n")
            print("Annotations saved to annotations.txt")
        elif key == ord("c"):
            # Clear annotations
            annotations.clear()
            annotated_image = image.copy()
            print("Annotations cleared")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    PathNames = r"D:\02_Lectures\2024_2nd\Lecture_Materials\SW_Dev\Project\val2017\val2017"
    segment_image(PathNames + "//000000000285.jpg")
