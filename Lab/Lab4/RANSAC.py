import numpy as np 
import os 
import cv2
import matplotlib.pyplot as plt 
import random

def RANSAC(img, kp1, kp2, matches, N = 1, P = 2): 

    kp1_cor = np.array([kp1[match.queryIdx].pt for match in matches])
    kp2_cor = np.array([kp2[match.trainIdx].pt for match in matches])
    x1, y1 = kp1_cor[:,0], kp1_cor[:,1]
    x2, y2 = kp2_cor[:,0], kp2_cor[:,1]

    inline_max = 0
    for i in range(N):
        # P matches at random from the total set of matches 
        random_matches = random.sample(range(0, len(x1)), P)
        x_p, y_p, x_t_p, y_t_p = x1[random_matches], y1[random_matches], x2[random_matches], y2[random_matches]
        
        # create matrix A and vector b 
        A = np.zeros((2*P,6))
        b = np.zeros((2*P,1))
        index = 0 
        for i in range(P):
            A[index,:] = [x_p[i],y_p[i], 0, 0, 1, 0]
            A[index+1,:] = [0, 0, x_p[i],y_p[i], 0, 1]
            b[index,:] = x_t_p[i]
            b[index+1,:] = y_t_p[i]
            index += 2

        X = np.dot(np.linalg.inv(A),b) 
    
        # Transforme all matches
        x_trans = np.zeros_like(x1)
        y_trans = np.zeros_like(y1)
        for i in range(len(x1)): 
            A_mat = np.array([[x1[i],y1[i], 0, 0, 1, 0], [0, 0, x1[i],y1[i], 0, 1]])
            trans = np.dot(A_mat, X)
            x_trans[i], y_trans[i]  = trans[0], trans[1]

        distance = np.sqrt((x_trans-x2)**2 + (y_trans-y2)**2)

        print(np.sum(distance < 10))

        if np.sum(distance < 10) > inline_max: 
            inline_max = np.sum(distance < 10)
            #I = distance<10 
            best_X = X
    
    return best_X

def transformation(img, X): 
    
    # Converting images to gray scale if necessary 
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
    trans_img = np.zeros_like(img)

    for r in range(trans_img.shape[0]): 
        for c in range(trans_img.shape[1]): 
            trans_img[r][c] = 255 
            A_mat = np.array([[img[r],img[c], 0, 0, 1, 0], [0, 0, img[r],img[c], 0, 1]])
            trans = np.dot(A_mat, X)
            i,j  = trans[0], trans[1]
            i = i.astype(np.int32)
            j = j.astype(np.int32)

    return trans_img
     

def built_in(img):
    # open CV transformation 
    rows,cols, _ = img.shape
    pts1 = np.float32([[300,200],[600,200],[300,400]])
    pts2 = np.float32([[260,200],[540,160],[340,400]])
    # OpenCV 
    M = cv2.getAffineTransform(pts1,pts2)
    builtin_img = cv2.warpAffine(img,M,(cols,rows))
    
    return builtin_img

def plot_images(original_img, target_img, cust_img , builtin_img): 
    h, w, _ = original_img.shape

    fig, ax = plt.subplots(2,2,)
    ax[0, 0].set_title('Original Image')
    ax[0, 0].scatter([300, 600, 300],[200, 200, 400], color='red')
    ax[0, 0].scatter([300, 560, 340],[200, 180, 400], color='blue')
    ax[0, 0].imshow(original_img)
    ax[0, 0].axis(False)
    ax[0, 1].set_title('Target Image')
    ax[0, 1].imshow(target_img)
    ax[0, 1].axis(False)
    ax[1, 0].set_title('Custom Transformation')
    ax[1, 0].imshow(cust_img, cmap = 'gray')
    ax[1, 0].axis(False)
    ax[1, 1].set_title('Opencv Transformation')
    ax[1, 1].scatter([300, 600, 300],[200, 200, 400], color='red')
    ax[1, 1].imshow(builtin_img)
    ax[1, 1].axis(False)
    plt.show()
