import tensorflow as tf
import numpy as np
from code import plots
import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.load_model("fashion_mnist_model")

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

img = test_images[6262]
plt.imshow(img)

img = (np.expand_dims(img, 0))
prediction_single = probability_model.predict(img)

plt.figure()
plots.plot_value_array(1, prediction_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

index = np.argmax(prediction_single[0])
print("This is: ", class_names[index])
