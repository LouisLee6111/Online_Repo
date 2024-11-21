# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:18:07 2024

@author: UOU
"""

import cv2, os
from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

folder_path = r"D:\03_Lectures\2024_2nd\Lecture_Materials\SW_Dev\SW_Dev\test_folder\Image_dataset"
file_names = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

results = model(folder_path+'\\'+file_names[0])

# Load the image using cv2
image = cv2.imread(folder_path+'\\'+file_names[0])

# Draw bounding boxes and labels on the image
for result in results:  # Iterate through results (one per detection)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        label = box.cls[0]  # Class label index
        label_text = f"{model.names[int(label)]} {confidence:.2f}"
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw the label
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('YOLO Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
