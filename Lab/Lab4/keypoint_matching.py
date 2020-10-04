import cv2
import numpy as np 
import os
import random
import matplotlib.pyplot as plt

def keypoint(img1, img2, sigma =1.6): 

    # Converting images to gray scale if necessary 
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(img1, (7,7), 6)
    #img1 = cv2.Canny(img1, 100,200 )

    img2 = cv2.GaussianBlur(img2, (7,7), 6)
    #img2 = cv2.Canny(img2, 100,200 )

    # finding key points 
    print('Finding interest points in photos...')
    sift = cv2.SIFT_create()

    """
    pip install opencv-python==4.0.44
    pip install opencv-contrib-python==4.0.44
    """

    #sift = cv2.SIFT_create(sigma=sigma)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # feature matching 
    print('Finding matching interest points....')
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)

    # todo: find the best matches 
    
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)

    return kp1, kp2, matches

def plot_matched_points(img1, img2, kp1, kp2, matches, k=10): 

    sample_matches = random.choices(matches, k=k)
    match = []
    # for m, n in sample_matches:
    #     match.append([m])

    for match in matches:  
        print(match.distance)

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, sample_matches, img2, flags=2)
    # matched_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, match,img2, flags=2)
    scale = .6
    matched_img = cv2.resize(matched_img, (np.int32(img2.shape[1]*2*scale), np.int32(img2.shape[0]*scale)))
    # cv2.imshow("Draw Matches", matched_img)
    # cv2.waitKey(0)
    plt.imshow(matched_img)
    plt.axis('off')
    plt.title('Matched points between two images')
    plt.show()

if __name__ == "__main__":
    
    path = os.path.dirname(os.path.abspath(__file__))
    img1_path = path + '/boat1.pgm'
    img2_path = path + '/boat2.pgm'
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # find key points of given images
    kp1, kp2, matches  = keypoint(img1, img2)
    
    # Draw matches ked points 
    plot_matched_points(img1, img2, kp1, kp2, matches)

