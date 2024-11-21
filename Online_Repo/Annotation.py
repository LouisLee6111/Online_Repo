# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:35:23 2024

@author: UOU
"""

import cv2, os
import numpy as np

# Parameters for drawing
first_click = True  # True if waiting for the first click
ix, iy = -1, -1  # Initial x, y coordinates of the region
rectangles = []

# Mouse callback function to draw rectangles
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, first_click

    if event == cv2.EVENT_LBUTTONDOWN:
        if first_click:
            # First click, set initial point
            ix, iy = x, y
            first_click = False
        else:
            # Second click, set the opposite corner and save the rectangle
            w, h = x - ix, y - iy
            rectangles.append((ix, iy, w, h))
            first_click = True

# Function to display the image and collect annotations
def segment_images_in_folder(folder_path, output_file="annotations.txt"):
    file_names = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    
    if not file_names:
        print("No images found in the folder!")
        return

    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found: {file_name}")
            continue

        # Create a clone of the image for annotation display
        annotated_image = image.copy()
        cv2.namedWindow("Image Segmentation")
        cv2.setMouseCallback("Image Segmentation", draw_rectangle)

        while True:
            # Show the annotations on the cloned image
            temp_image = annotated_image.copy()
            for rect in rectangles:
                x, y, w, h = rect
                cv2.rectangle(temp_image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

            # Display the image with annotations
            cv2.imshow("Image Segmentation", temp_image)
            
            # Press 's' to save annotations, 'c' to clear, and 'q' to move to the next image
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                # Save annotations to file
                with open(output_file, "a") as f:
                    for rect in rectangles:
                        f.write(f"{file_name}: {rect}\n")
                print(f"Annotations saved for {file_name} to {output_file}")
            elif key == ord("c"):
                # Clear annotations
                rectangles.clear()
                annotated_image = image.copy()
                print("Annotations cleared")
            elif key == ord("q"):
                # Quit the entire process
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                # Move to the next image
                rectangles.clear()
                break

        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    folder_path = r"D:\03_Lectures\2024_2nd\Lecture_Materials\SW_Dev\SW_Dev\test_folder\Image_dataset"
    segment_images_in_folder(folder_path)
