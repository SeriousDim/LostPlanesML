import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd

tf.config.list_physical_devices('GPU')

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop("Age")

abalone_features = np.array(abalone_features)

normalize = preprocessing.Normalization()
normalize.adapt(abalone_features)
model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(),
              optimizer = tf.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(abalone_features, abalone_labels, epochs=30)


