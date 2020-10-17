import numpy as np 
import cv2 
# from stl10_input import *
from sklearn.cluster import KMeans
# from __future__ import print_function
import sys
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler


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

    descriptors = []

    sift = cv2.xfeatures2d.SIFT_create()

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        if des is not None:
            for d in des:
                descriptors.append(d)


        #img1 = cv2.drawKeypoints(gray, kp, gray)
        #plt.imshow(img1)
        #plt.show()

    descriptors = np.asarray(descriptors)

    return descriptors

# 3 using k-means clustering to generate visual vocabulary
def visual_vocab(n_words, descriptors):
    kmeans = KMeans(n_clusters=n_words)  #n_clusters is the number of words we want in dictionary
    kmeans.fit(X=descriptors)
    visual_words = kmeans.cluster_centers_
    return visual_words 

def constructHisto(images, words):
    sift = cv2.xfeatures2d.SIFT_create()
    image_histos = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        #print('LEN DES', len(des))
        #for d in des:
        #    print('LEN D', len(d))
        # For each extracted feature vector, compute its nearest neighbor in the dictionary created in Step #2 — this is normally accomplished using the Euclidean Distance.
        closest_words = []

        for d in des:
            closest_word = words[0]
            closest_distance = np.linalg.norm(d-words[0])
            for w in words:
                euclidean_dist = np.linalg.norm(d-w)
                if euclidean_dist < closest_distance:
                    closest_word = w
                    closest_distance = euclidean_dist
            closest_words.append(closest_word)
        image_histos.append(closest_words)
        #plt.hist(closest_words, bins=len(words))
        #plt.show()

    image_histos = np.array(image_histos, dtype=object)
    return image_histos


def getFeaturesVector(image_histos, words):
    image_features = []
    for histo in image_histos:
        feature_vec = []
        for word in words:
            count = 0
            for histo_word in histo:
                if np.array_equal(word, histo_word):#(histo_word == word).all():
                    count += 1
            feature_vec.append(count)
        image_features.append(feature_vec)
    #image_features = np.asarray(image_features).reshape(-1, 1)
    return image_features


def trainSVC(image_histos, labels):
    scaler = StandardScaler().fit(image_histos)
    image_histos = scaler.transform(image_histos)
    clf = SVC()
    clf.fit(image_histos, labels)
    return clf


def sortClasses(images, labels):
    airplanes1 = []
    birds2 = []
    ships3 = []
    horses4 = []
    cars5 = []
    for label in labels:
        if label == 1:
            airplanes1.append(images[np.where(labels == label)])
        elif label == 2:
            birds2.append(images[np.where(labels == label)])
        elif label == 3:
            ships3.append(images[np.where(labels == label)])
        elif label == 4:
            horses4.append(images[np.where(labels == label)])
        elif label == 5:
            cars5.append(images[np.where(labels == label)])
    return airplanes1, birds2, ships3, horses4, cars5

def main(): 
    download_and_extract()

    with open(DATA_PATH) as f:
        image = read_single_image(f)
        # plot_image(image)
    
    images = read_all_images(DATA_PATH)
    labels = read_labels(LABEL_PATH)
    print(type(images[0]), type(labels))
    print(np.unique(labels))

    image_new = []
    labels_new = []

    for i in range(len(labels)):
        if labels[i] == 5:
            cats.append(image[i])
            image_new.append(images[i])
            labels_new.append(labels[i])
    """
    for i in range(len(image_new)):
        plt.imshow(image_new[i])
        plt.title(labels_new[i])
        plt.show()
    """

    print('LEN IMAGES', len(image_new))

    #airplanes1, birds2, ships3, horses4, cars5 = sortClasses(images, labels)
    ''''
    i = 0
    for plane in airplanes1:
        plt.imshow(plane)
        plt.title('plane')
        plt.show()
        i += 1
        if i > 7:
            break
    '''

    plt.imshow(images[39])
    plt.title(labels[39])
    plt.show()

    descriptors = detect_sift(images[0:40])

    words = visual_vocab(1000, descriptors)      # try 400, 1000, 4000

    print(len(words))

    # convert each image to its histogram representation

    image_histos = constructHisto(images[0:40], words)

    image_features = getFeaturesVector(image_histos[0:40], words)

    print('IF', len(image_features))
    print('IF0', len(image_features[0]))

    clf = trainSVC(image_features[0:39], labels[0:39])

    print('Actual Class', labels[39])
    print('Predicted Class', clf.predict([image_features[39]]))






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

Step #5: Vector quantization

Given an arbitrary image (whether from our original dataset or not), we can quantify and abstractly represent the image using our bag of visual words model by applying the following process:

    Extract feature vectors from the image in the same manner as Step #1 above.
    For each extracted feature vector, compute its nearest neighbor in the dictionary created in Step #2 — this is normally accomplished using the Euclidean Distance.
    Take the set of nearest neighbor labels and build a histogram of length k (the number of clusters generated from k-means), where the i‘th value in the histogram is the frequency of the i‘th visual word. This process in modeling an object by its distribution of prototype vectors is commonly called vector quantization.

https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/
https://medium.com/@aybukeyalcinerr/bag-of-visual-words-bovw-db9500331b2f
https://machinelearningknowledge.ai/image-classification-using-bag-of-visual-words-model/
'''


