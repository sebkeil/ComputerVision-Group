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
import pandas as pd

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

# path to the test images and labels

DATA_PATH_TEST = './data/stl10_binary/test_X.bin'
LABEL_PATH_TEST = './data/stl10_binary/test_y.bin'


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

        ### COMMENT OUT TO SHOW KEYPOINTS ###
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

def construct_bow_rep(images, words):
    sift = cv2.xfeatures2d.SIFT_create()
    images_as_bow = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        # For each extracted feature vector, compute its nearest neighbor in the dictionary created in Step #2 —
        # this is normally accomplished using the Euclidean Distance.

        closest_words = []      # contains the nearest neighbors of all the descriptor

        if des is not None:
            for d in des:
                # initialize closest word to the first word, closest distance to the first distance
                closest_word = words[0]
                closest_distance = np.linalg.norm(d-words[0])
                # go through all the words and update the closest one (using euclidean distance)
                for w in words:
                    euclidean_dist = np.linalg.norm(d-w)
                    if euclidean_dist < closest_distance:
                        closest_word = w
                        closest_distance = euclidean_dist
                closest_words.append(closest_word)
            images_as_bow.append(closest_words)
            #plt.hist(closest_words, bins=len(words))
            #plt.show()

    images_as_bow = np.array(images_as_bow, dtype=object)
    return images_as_bow


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


def trainSVC(bin, neg_bin):
    positives = np.ones((len(bin)), dtype=float)
    negatives = np.zeros((len(neg_bin)), dtype=float)
    data = np.concatenate((bin, neg_bin))
    labels = np.concatenate((positives, negatives))
    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)
    clf = SVC(probability=True)
    clf.fit(data, labels)
    return clf


def sortClasses(images, labels, bin_size):

    bin_1 = []
    bin_2 = []
    bin_3 = []
    bin_4 = []
    bin_5 = []

    for i in range(len(labels)):
        if labels[i] == 1 and len(bin_1) < bin_size:
            bin_1.append(images[i])
        elif labels[i] == 2 and len(bin_2) < bin_size:
            bin_2.append(images[i])
        elif labels[i] == 3 and len(bin_3) < bin_size:
            bin_3.append(images[i])
        elif labels[i] == 4 and len(bin_4) < bin_size:
            bin_4.append(images[i])
        elif labels[i] == 5 and len(bin_5) < bin_size:
            bin_5.append(images[i])

    return bin_1, bin_2, bin_3, bin_4, bin_5

def filter_images(images, labels):
    image_sub = []
    label_sub = []
    for i in range(len(labels)):
        if labels[i] == 1 or labels[i] == 2:
            image_sub.append(images[i])
            label_sub.append(labels[i])
        elif labels[i] == 9:
            image_sub.append(images[i])
            label_sub.append(3)
        elif labels[i] == 7:
            image_sub.append(images[i])
            label_sub.append(4)
        elif labels[i] == 3:
            image_sub.append(images[i])
            label_sub.append(5)
    return image_sub, label_sub


def plot_histograms(image_features, labels_sub, words):
    category_dict = {
        1: ' 1 - Airplane',
        2: ' 2 - Bird',
        3: ' 3 - Ship',
        4: ' 4 - Horse',
        5: ' 5 - Car'}

    w_bins = [w for w in range(len(words))]

    for i in range(len(image_features)):
        plt.bar(w_bins, height=image_features[i])
        plt.title('Category: {}'.format(category_dict[labels_sub[i]]))
        plt.xlabel('Visual Word #')
        plt.ylabel('Frequency')
        plt.show()
        if i >= 8:
            break

def getNegatives(images, labels, bin_size):

    bin_1 = []
    bin_2 = []
    bin_3 = []
    bin_4 = []
    bin_5 = []

    for i in range(len(labels)):
        if labels[i] != 1 and len(bin_1) < bin_size:
            bin_1.append(images[i])
        elif labels[i] != 2 and len(bin_2) < bin_size:
            bin_2.append(images[i])
        elif labels[i] != 3 and len(bin_3) < bin_size:
            bin_3.append(images[i])
        elif labels[i] != 4 and len(bin_4) < bin_size:
            bin_4.append(images[i])
        elif labels[i] != 5 and len(bin_5) < bin_size:
            bin_5.append(images[i])

    return bin_1, bin_2, bin_3, bin_4, bin_5

def make_predictions(image_feature_test, labels_sub_test, clf1, clf2, clf3, clf4, clf5):
    bin1_preds = []
    #bin1_preds_perc = []
    bin2_preds = []
    #bin2_preds_perc = []
    bin3_preds = []
    #bin3_preds_perc = []
    bin4_preds = []
    #bin4_preds_perc = []
    bin5_preds = []
    #bin5_preds_perc = []

    for i in range(len(image_feature_test)):
        image_pred = []
        # predict with all 5 classifiers
        pred1 = {'class': 1, 'prob': clf1.predict_proba([image_feature_test[i]])[0][1]}
        pred2 = {'class': 2, 'prob': clf2.predict_proba([image_feature_test[i]])[0][1]}
        pred3 = {'class': 3, 'prob': clf3.predict_proba([image_feature_test[i]])[0][1]}
        pred4 = {'class': 4, 'prob': clf4.predict_proba([image_feature_test[i]])[0][1]}
        pred5 = {'class': 5, 'prob': clf5.predict_proba([image_feature_test[i]])[0][1]}

        image_pred.append(pred1)
        image_pred.append(pred2)
        image_pred.append(pred3)
        image_pred.append(pred4)
        image_pred.append(pred5)

        # rank the sorting for each image
        image_pred = np.asarray(sorted(image_pred, key=lambda x: x['prob'], reverse=True))

        # assign to the classification

        #image_rank = [pred['class'] for pred in image_pred]

        #image_rank = np.asarray(image_rank)
        # image_rank = np.reshape(image_rank, (5, 1))

        if labels_sub_test[i] == 1:
            #bin1_preds_perc.append(image_pred[0]['prob'])
            bin1_preds.append(image_pred)
        elif labels_sub_test[i] == 2:
            #bin2_preds_perc.append(image_pred[0]['prob'])
            bin2_preds.append(image_pred)
        elif labels_sub_test[i] == 3:
            #bin3_preds_perc.append(image_pred[0]['prob'])
            bin3_preds.append(image_pred)
        elif labels_sub_test[i] == 4:
            #bin4_preds_perc.append(image_pred[0]['prob'])
            bin4_preds.append(image_pred)
        elif labels_sub_test[i] == 5:
            #bin5_preds_perc.append(image_pred[0]['prob'])
            bin5_preds.append(image_pred)

    #bin1_preds = sorted(bin1_preds, key=lambda k: k['prob'] if k['class']==1, reverse=False)
    print(bin1_preds)

    return bin1_preds, bin2_preds, bin3_preds, bin4_preds, bin5_preds #, bin1_preds_perc, bin2_preds_perc, bin3_preds_perc, bin4_preds_perc, bin5_preds_perc

def mean_avg_precision(bin, class_nr):

    def f_c(i, bin, class_nr):
        counter = 0
        if bin[i][0]['class'] == class_nr:
            for j in range(i):
                #dict = bin[j][0]
                #print(dict)
                #print(dict['class'])
                if bin[j][0]['class'] == class_nr:
                    counter += 1
        return counter

    n = len(bin)
    m = len(bin)/5

    sum = 0
    for i in range(1, n):
        temp = f_c(i, bin, class_nr) / i
        sum += temp

    res = (1/m) * sum
    return res


def main():
    download_and_extract()

    with open(DATA_PATH) as f:
        image = read_single_image(f)
        # plot_image(image)

    print("Reading in training images...")
    images = read_all_images(DATA_PATH)
    labels = read_labels(LABEL_PATH)
    print("Done! Read in {} images.".format(len(images)))

    print("Filtering out images and renaming classes...")
    # get subset of half the images from all categories
    images_sub, labels_sub = filter_images(images, labels)
    print("Done! Filtered out {} images, {} labels.".format(len(images_sub), len(labels_sub)))      # should be 2500

    # lets seperate those images into two subsets, first one we'll use for building the visual vocab, second one for training the classifiers

    images_sub_1 = images_sub[:100]
    labels_sub_1 = labels_sub[:100]

    images_sub_2 = images_sub[1000:]
    labels_sub_2 = labels_sub[1000:]

    print("Batch 1: {} images, Batch 2: {} images.".format(len(images_sub_1), len(images_sub_2)))


    print('Building descriptors from Batch 1...')
    # get descriptors and build visual vocabulary
    descriptors = detect_sift(images_sub_1)
    print('Done! There are {} descriptors in total now'.format(len(descriptors)))

    print('Generating Visual Vocabulary...')
    words = visual_vocab(1000, descriptors)      # try 400, 1000, 4000
    print('Done! Generated {} visual words from descriptors.'.format(len(words)))

    print('Converting each image into its bag of words representation...')
    # convert each image to its histogram representation
    images1_as_bow = construct_bow_rep(images_sub_1, words)
    print('Done! There are {} images in their BoW representation now.'.format(len(images1_as_bow)))

    print('Computing frequency vectors for the images..')
    image_features1 = getFeaturesVector(images1_as_bow, words)
    print('Done! We have {} image vectors now, with each vector consisting of {} features'.format(len(image_features1), len(image_features1[0])))

    plot_histograms(image_features1, labels_sub_1, words)

    print('Computing BoW representations for images from Batch 2...')
    images2_as_bow =  construct_bow_rep(images_sub_2, words)
    image_feature2 = getFeaturesVector(images2_as_bow, words)
    print('Done! We have  {} image vectors now, with each vector consisting of {} features'.format(len(image_features1), len(image_features1[0])))


    print('Sorting remaining images into bins...')
    airplanes1, birds2, ships3, horses4, cars5 = sortClasses(image_feature2, labels_sub_2, bin_size=50)
    print('Done! Sorted remaining images into 5 bins of lengths: {}, {}, {}, {}, {}'.format(len(airplanes1), len(birds2), len(ships3), len(horses4), len(cars5)))

    print('Getting negative examples and sorting them')
    neg_1, neg_2, neg_3, neg_4, neg_5 = getNegatives(image_feature2, labels_sub_2, bin_size=200)
    print('Done! Sorted negative images into 5 bins of lengths: {}, {}, {}, {}, {}'.format(len(neg_1), len(neg_2),len(neg_3), len(neg_4),len(neg_5)))

    print('Training classifiers...')
    clf1 = trainSVC(airplanes1, neg_1)
    clf2 = trainSVC(birds2, neg_2)
    clf3 = trainSVC(ships3, neg_3)
    clf4 = trainSVC(horses4, neg_4)
    clf5 = trainSVC(cars5, neg_5)
    print('Done! Trained all classifiers.')


    print('Actual Class', labels_sub[5])
    print('Predicted by Classifier 1', clf1.predict_proba([image_features1[5]])[0][1])
    print('Classes', clf1.classes_)
    print('Predicted by Classifier 2', clf2.predict_proba([image_features1[5]]))
    print('Predicted by Classifier 3', clf3.predict_proba([image_features1[5]]))
    print('Predicted by Classifier 4', clf4.predict_proba([image_features1[5]]))
    print('Predicted by Classifier 5', clf5.predict_proba([image_features1[5]]))

    print("Reading in testing images...")
    images_test = read_all_images(DATA_PATH_TEST)
    labels_test = read_labels(LABEL_PATH_TEST)
    print("Done! Read in {} images.".format(len(images)))

    print("Filtering out images and renaming classes...")
    # get subset of half the images from all categories
    images_sub_test, labels_sub_test = filter_images(images_test, labels_test)
    print("Done! Filtered out {} images, {} labels.".format(len(images_sub), len(labels_sub)))      # should be 2500

    # get subset just for testing it out

    #images_sub_test = images_sub_test[:1000]
    #labels_sub_test = labels_sub_test[:1000]

    print('Computing BoW representations for test set images...')
    images_test_as_bow = construct_bow_rep(images_sub_test, words)
    image_feature_test = getFeaturesVector(images_test_as_bow, words)
    print('Done! We have  {} image vectors now, with each vector consisting of {} features'.format(len(image_features1), len(image_features1[0])))

    bin1_preds, bin2_preds, bin3_preds, bin4_preds, bin5_preds = make_predictions(image_feature_test, labels_sub_test, clf1, clf2, clf3, clf4, clf5)


    "------------ Top 5 Ranks ---------------------"
    print('BIN1',bin1_preds[:20])
    print('BIN2', bin2_preds[:20])
    print('BIN3', bin3_preds[:20])
    print('BIN4', bin4_preds[:20])
    print('BIN5', bin5_preds[:20])
    print('-----------------------------------------')

    print("------------ Bottom 5 Ranks ---------------------")
    print('BIN1', bin1_preds[len(bin1_preds)-20:])
    print('BIN2', bin2_preds[len(bin2_preds)-20:])
    print('BIN3', bin3_preds[len(bin3_preds)-20:])
    print('BIN4', bin4_preds[len(bin4_preds)-20:])
    print('BIN5', bin5_preds[len(bin5_preds)-20:])
    print('-----------------------------------------')


    map1 = mean_avg_precision(bin1_preds, 1)
    map2 = mean_avg_precision(bin2_preds, 2)
    map3 = mean_avg_precision(bin3_preds, 3)
    map4 = mean_avg_precision(bin4_preds, 4)
    map5 = mean_avg_precision(bin5_preds, 5)

    overall_map = (map1 + map2 + map3 + map4 + map5) / 5

    print('MAP1', map1)
    print('MAP2', map2)
    print('MAP3', map3)
    print('MAP4', map4)
    print('MAP5', map5)
    print('Overall', overall_map)

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


"""
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
"""

# bin_1_preds = sorted(bin1_preds, key=