import numpy as np
import cv2

img1 = cv2.imread('boat1.pgm')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp, des = sift.detectAndCompute(gray1, None)

img1 = cv2.drawKeypoints(gray1, kp, img1)

cv2.imshow('kp-image', img1)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(des)

#cv2.imwrite('sift_keypoints.jpg', img1)


