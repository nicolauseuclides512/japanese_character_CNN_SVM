import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

letter_data = tf.data.experimental.CsvDataset(
    '/home/nicolaus/PycharmProjects/thesis_program/japanese_letter_recognition/Kana_Letter.csv',
    record_defaults=[tf.string, tf.string],
    select_cols=[0, 1],
    field_delim=",",
    header=True
)

for data in letter_data:
    tf.print(data)  # Print : {'Name': [Wii Sports], 'Platform': [Wii], 'Year': [2006]}
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(168, 168, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
# model.summary()
