import tensorflow as tf
import numpy as np
from prepare_data_from_csv import *

"""

См. файл fahsion_mnist_conv.py

"""

print(tf.__version__)

tf.config.list_physical_devices('GPU')

imgs, names = read_imgs_for_answer()

class_names = ['None', 'Plane']

imgs = imgs / 255.0

model = tf.keras.models.load_model("lost_planes_model")

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

data = []
imgs = imgs.reshape((1000, 20, 20, 3))

prediction_single = probability_model.predict(imgs)
c = 0
for p in prediction_single:
    index = np.argmax(p)
    print(names[c], class_names[index])
    c += 1

"""
c = 0
for img in imgs:
    img = (np.expand_dims(img, 0))
    prediction_single = probability_model.predict(img)
    print(names[c], class_names[index])
    c += 1
    """
