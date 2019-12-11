import tensorflow as tf
import matplotlib.pyplot as plot
import random 
import numpy as np
import cv2
import os
import pickle


def extract_data():
    # directory where taining data is stored
    DIR = 'C:/Users/josh/Projects/PlantSeedlingClassifier/train/'

    # all classes 
    CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common Wheat',
                    'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',
                    'Small-flowered Cranesbill', 'Sugar Beet']

    # standard img size to be used for all images
    IMG_SIZE = 100

    train_data = []

    # iterate through all classes 
    for cat in CATEGORIES:
        path = os.path.join(DIR, cat)
        label = CATEGORIES.index(cat)
        for img in os.listdir(path):
            # change to image to be greyscale
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

            # resize image to be 100x100
            resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            train_data.append([resized, label])

    # randomize order of the data 
    random.shuffle(train_data)

    X = []
    Y = []

    # fill the lists with the images and labels
    for img, label in train_data:
        X.append(img)
        Y.append(label)

    # reshape array to be the same shape as the images
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    # pickle and dump data
    pickled_x = open('trainpickle/imgs/imgs.pickle', 'wb')
    pickle.dump(X, pickled_x)
    pickled_x.close()

    pickled_y = open('trainpickle/labels/labels.pickle', 'wb')
    pickle.dump(X, pickled_y)
    pickled_y.close()

extract_data()