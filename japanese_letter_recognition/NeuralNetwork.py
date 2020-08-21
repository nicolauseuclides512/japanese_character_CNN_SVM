from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import pydot
import graphviz
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D


def convnetModel(widht, height, data):
    model = Sequential()
    model.add(Conv2D(1, (3, 3), activation='relu', input_shape=(widht, height, 1)))
    model.add(AveragePooling2D())
    model.summary()
    yhat = model.predict(data)
    for r in range(yhat.shape[1]):
        # print each column in the row
        print([yhat[0, r, c, 0] for c in range(yhat.shape[2])])

def splitData(data_with_label):
    data_train, data_test, label_train, labe_test = train_test_split(data_with_label[0], data_with_label[1],
                                                                     test_size=0.33)
    print(data_train)
