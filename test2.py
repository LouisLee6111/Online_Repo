# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:11:29 2024

@author: cic
"""
import cv2

# 이미지 경로
image_path = '/mnt/data/000000005529.jpg'

# 이미지 로드
image = cv2.imread(image_path)

# HOG 기반 사람 탐지기 설정
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 사람 탐지 수행
(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(16, 16), scale=1.05)

# 탐지된 사람들에 대한 결과 저장
detected_people = []
for (x, y, w, h) in rects:
    detected_people.append((x, y, w, h))

# 탐지 결과 출력
print("탐지된 사람 좌표 및 크기 (x, y, 너비, 높이):")
for person in detected_people:
    print(person)

# 결과 시각화
for (x, y, w, h) in detected_people:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 결과 이미지 저장 및 표시
output_path = "/mnt/data/detected_people.jpg"
cv2.imwrite(output_path, image)
print(f"탐지 결과를 저장했습니다: {output_path}")