import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plot
import random 
import numpy as np
import cv2
import os
import pickle


def initialize_model():
    # create model
    model = Sequential()

    # add first convolutional layer with a 3x3 kernel
    model.add(Conv2D(64, 3, 3, input_shape=(100, 100, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # create feed forward part of neural network
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(12))

    # output activation must be probability for cross-entropy
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train():
    # get model
    model = initialize_model()
    
    # open pickled data
    train_imgs = pickle.load(open('trainpickle/imgs/imgs.pickle', 'br'))/255.0 # normalize images
    train_labels = pickle.load(open('trainpickle/labels/labels.pickle', 'br'))

    # labels must be put into an numpy array
    train_labels = np.array(train_labels)

    # train and save model
    model.fit(train_imgs, train_labels)

    model.save('superseedlingseeer.model')


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
    pickle.dump(Y, pickled_y)
    pickled_y.close()

#extract_data()
train()