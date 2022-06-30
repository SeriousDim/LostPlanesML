import tensorflow as tf
import numpy as np
from prepare_data_from_csv import *
from sklearn.metrics import classification_report
from pyimagesearch import config
import pickle

"""

См. файл fahsion_mnist_conv.py

"""

print(tf.__version__)

tf.config.list_physical_devices('GPU')

print("[INFO] reading data...")
imgs, names = read_imgs_for_answer()

class_names = ['None', 'Plane']

imgs = imgs / 255.0

model = tf.keras.models.load_model("lost_planes_model")

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

imgs = imgs.reshape((1000 + 100000, 20, 20, 3))

print("[INFO] predicting...")
predIdxs = model.predict(imgs)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

data = []
c = 0
for p in predIdxs:
    data.append((p, names[c]))
    c += 1

# Write CSV file
print("[INFO] building CSV...")
with open("answer.csv", "wt", newline='') as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["sign", "filename"])  # write header
    writer.writerows(data)
