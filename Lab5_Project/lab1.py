import numpy as np 
import cv2 
# from stl10_input import *
import sklearn
# from __future__ import print_function
import sys
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import random

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib
else:
    import urllib

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = './data/stl10_binary/train_X.bin'

# path to the binary train file with labels
LABEL_PATH = './data/stl10_binary/train_y.bin'

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
        print('folder created')
    else:
       print(f'{dest_directory} exists') 
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    print(filepath)
    # tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def read_single_image(image_file):
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    image = np.transpose(image, (2, 1, 0))
    return image
def plot_image(img): 
    plt.imshow(img)
    plt.show()

def read_all_images(path_to_data):


    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        
        images = np.reshape(everything, (-1, 3, 96, 96))

        images = np.transpose(images, (0, 3, 2, 1))
        return images

def read_labels(path_to_labels):

    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


# get function to draw a subset (all categories)
def draw_subset(images):
    k = len(images)/2
    images = random.sample(images, k)
    return images


# 2 - sift detector
def detect_sift(images):

    keypoints = []
    descriptors = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
        # cv2.drawKeypoints(gray, kp, img)
        # cv2.waitKey(0)
        
    return keypoints, descriptors


# 3 using k-means clustering to generate visual vocabulary
def visual_vocab(n_words, descriptors):
    kmeans = KMeans(n_clusters=n_words, random_state=0)  #n_clusters is the number of words we want in dictionary
    kmeans.fit(descriptors)
    visual_words = kmeans.cluster_centers_
    return visual_words 


def main(): 
    download_and_extract()

    with open(DATA_PATH) as f:
        image = read_single_image(f)
        # plot_image(image)
    
    images = read_all_images(DATA_PATH)

    print(images.shape)

    labels = read_labels(LABEL_PATH)
    print(labels.shape)

    plt.imshow(images[0])
    plt.title(labels[0])
    #plt.show()

    kps, des = detect_sift(images[3:])

    print(len(kps))
    print(len(des))


    




if __name__ == '__main__':
    main()
    
# todo: 1. 

'''
Steps to take:

1. Read in training images from train X.bin (probably images ),train y.bin (probably classifications)
    > use subset for testing if our algo works
    > Take a subset (maximum half) of all training images

    
2. Extract SIFT descriptor from training datasets and show ﬁve images with diﬀerent classes
    > we need to store the information for each image, keep track of classifications

3. Using the SIFT descriptors we implement k-means clustering (e.g. using sklearn library or cv2)
 
    > we store this as our visual vocabulary

4.  Then, take the rest of the training images to calculate visual dictionary. ??
    Visual dictionary: find cluster of similar descriptors, same as vocabulary 
    > set cluster size to different values (400, 1000, 4000)

5.  Represent each image by a histogram of its visual words
    > using matplotlib
    > use at least 50 images per class
    > If you use the default setting, you should have 50 histograms of size 400
    > obtain histograms of visual words for images from other classes, again about 50 images per class, as negative examples
    > you will have 200 negative examples
    
6. train a Support Vector Machine (SVM) classiﬁer on each image class (5 in total)
    > using sklearn
    > repeat for each class
    > To classify a new image, calculate its visual words histogram as described in Section 2.4
    > use the trained SVM classiﬁer to assign it to the most probable object class

7. Evaluation
    > load the test set images
    > rank images based on each binary classifier (check ML slides)
    > you will have 5 lists of test images
    > use Mean Average Precision to measure system performance

    

Note: Installation (pip install opencv-python==3.4.2.17  pip install opencv-contrib-python==3.4.2.17)


'''



