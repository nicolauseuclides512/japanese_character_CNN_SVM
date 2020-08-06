import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def convnetModel(data, labels, widht, height):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(widht, height, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2))
    model.compile(metrics=['accuracy'])
    model.fit(data, labels, verbose=1)


def splitData(data_with_label):
    data_train, data_test, label_train, labe_test = train_test_split(data_with_label[0], data_with_label[1],
                                                                     test_size=0.33)
    print(data_train)