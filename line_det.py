import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("line_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([20, 100, 100])
higher_yellow = np.array([30, 255, 255])

mask_yellow = cv2.inRange(hsv, lower_yellow, higher_yellow)

mask_white = cv2.inRange(gray, 168, 255)
mask_final = cv2.bitwise_or(mask_white, mask_yellow)
masked_image = cv2.bitwise_and(gray, mask_final)

kernel = np.ones((3,3), np.uint8)
dilate = cv2.dilate(masked_image, kernel, iterations = 1)
erosion = cv2.erode(dilate, kernel, iterations = 1)

edges = cv2.Canny(erosion, 100,200)


cv2.imshow("Y&W Detect",masked_image)
cv2.imshow("Dilated and Eroded",erosion)
cv2.imshow("Edge detection",edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
