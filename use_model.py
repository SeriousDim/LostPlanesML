import cv2 as cv
import csv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

"""
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
"""

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.load_model("model1.model")

img = cv.imread("data/sorted_train/1/0a6af15e-4cdd-4a68-a5ff-762f37255670.png")
img = cv.resize(img, (32, 32), interpolation = cv.INTER_AREA)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

prediction = model.predict(np.array([img]) / 255.0)
index = np.argmax(prediction)

print(f"Prediction: {class_names[index]}")
