import tensorflow as tf
import cv2 as cv
import numpy as np
from prepare_data_from_csv import *

"""

См. файл fahsion_mnist_conv.py

"""

print(tf.__version__)

tf.config.list_physical_devices('GPU')

(train_images, train_labels) = load_train()
#(test_images, test_labels) = load_test()

class_names = ['None', 'Plane']
train_images = np.float32(train_images)

for i in range(len(train_images)):
    train_images[i] = cv.cvtColor(train_images[i], cv.COLOR_BGR2GRAY)

train_images = train_images / 255.0
#test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(20, 20, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

BATCH_SIZE = 32

# насчет этого смотри ссылку 1
train_images = train_images.reshape((31080, 20, 20, 1))
#test_images = test_images.reshape((4610, 20, 20, 3))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(1000).batch(BATCH_SIZE)
#test_dataset = tf.data.Dataset.from_tensor_slices(
    #(test_images, test_labels)).batch(BATCH_SIZE)

#model.fit(train_dataset, epochs=8)
model.fit(train_dataset, epochs=2)
# , steps_per_epoch=math.ceil(26470/BATCH_SIZE)

#test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
#test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
# , steps=math.ceil(4610/BATCH_SIZE)
model.save("lost_planes_model")

#print('\nTest accuracy: ', test_acc)
