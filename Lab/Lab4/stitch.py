import numpy as np
import cv2
import matplotlib.pyplot as plt

imgR = cv2.imread('right.jpg')
imgL = cv2.imread('left.jpg')


def stitch(imgR, imgL):

    img1 = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    # find the keypoints
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # match most similar features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=5)      # give out 5 best matches

    # select good matches based on threshold
    threshold = 0.7
    good_matches = []
    for match in matches:
        #print(match[0], match[1])
        if match[0].distance < threshold * match[1].distance:
            good_matches.append(match)
            #print(match)
    matches = np.asarray(good_matches)

    source = np.float32([kp1[match.queryIdx].pt for match in matches[:, 0]]).reshape(-1, 1, 2)
    nearest = np.float32([kp2[match.trainIdx].pt for match in matches[:, 0]]).reshape(-1, 1, 2)
    h_matrix, mask = cv2.findHomography(source, nearest, cv2.RANSAC, 5.0)

    warped = cv2.warpPerspective(imgR, h_matrix, (imgL.shape[1] + imgR.shape[1], imgL.shape[0]))

    warped[0:imgL.shape[0], 0:imgL.shape[1]] = imgL
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # chop off black edges
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    warped = warped[y:y + h, x:x + w]

    plt.imshow(warped)
    plt.show()


stitch(imgR, imgL)
