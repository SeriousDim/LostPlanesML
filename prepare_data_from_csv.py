import numpy as np
import cv2 as cv
import csv
import os

def load_train():
    path = "data/train"
    amount = 31080 # rows in train.csv
    size = (20, 20, 3)
    train_images = np.array([np.zeros(size) for i in range(amount)])
    train_labels = np.array([0 for i in range(amount)])
    with open('data/train.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        c = 0
        for row in spamreader:
            img = cv.imread(path + "/" + row[1] + ".png")
            train_images[c] = img
            train_labels[c] = int(row[0])
            c += 1
    return train_images, train_labels

def load_test():
    path = "/data/train"
    amount = 4610 # rows in my_train.csv
    size = (20, 20, 3)
    test_images = np.array([np.zeros(size) for i in range(amount)])
    test_labels = np.array([0 for i in range(amount)])
    with open('data/my_test.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        c = 0
        for row in spamreader:
            img = cv.imread(path + "/" + row[1] + ".png")
            test_images[c] = img
            test_labels[c] = int(row[0])
            c += 1
    return test_images, test_labels

def read_imgs_for_answer():
    path = "/data/test"
    amount = 1000  # rows in test.csv
    size = (20, 20, 3)
    result = np.array([np.zeros(size) for i in range(amount)])
    names = ['' for i in range(amount)]
    with open('data/test.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        c = 0
        for row in spamreader:
            img = cv.imread(path + "/" + row[0] + ".png")
            result[c] = img
            names[c] = row[0]
            c += 1
    return result, names
