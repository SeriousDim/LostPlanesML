import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

"""

1. https://stackoverflow.com/questions/43895750/keras-input-shape-for-conv2d-and-manually-loaded-images - reshape данных
2. https://www.youtube.com/playlist?list=PLfdVzZl6HHg9y9l6U5xUjqKS13rWoQPF4 - большой курс, к каждому видео прилагается статья на Хабре
3. https://habr.com/ru/post/454034/ - одна из статей курса
4. https://habr.com/ru/post/454986/ - статья по сверточные сети

"""

print(tf.__version__)

tf.config.list_physical_devices('GPU')

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

BATCH_SIZE = 32

# насчет этого смотри ссылку 1
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(60000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(BATCH_SIZE)

#model.fit(train_dataset, epochs=8)
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(60000/BATCH_SIZE))

#test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
test_loss, test_acc = model.evaluate(test_dataset, steps=math.ceil(60000/BATCH_SIZE))
model.save("fashion_mnist_model_2")

print('\nTest accuracy: ', test_acc)
