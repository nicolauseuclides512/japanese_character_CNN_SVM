import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, Ftrl, Nadam, RMSprop
from tensorflow.keras.losses import categorical_crossentropy, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, concatenate, Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU


def readDataDirectory():
    train_dir = 'potongan/train'
    test_dir = 'potongan/test'
    hiragana_train_dir = train_dir + "/Hiragana"
    katakana_train_dir = train_dir + "/Katakana"
    hiragana_test_dir = test_dir + "/Hiragana"
    katakana_test_dir = test_dir + "/Katakana"
    sample_submission = pd.read_csv('sample_submission.csv')
    return hiragana_train_dir, katakana_train_dir, hiragana_test_dir, katakana_test_dir, sample_submission

def dataLabeling():
    letter_labels = [
        [0, "A", "A"], [1, "Ba", "Ba"], [2, "Be", "Be"], [3, "Bi", "Bi"], [4, "Bo", "Bo"], [5, "Bu", "Bu"],
        [6, "Bya", "Bya"], [7, "Byo", "Byo"], [8, "Byu", "Byu"], [9, "Cha", "Cha"], [10, "Chi", "Chi"],
        [11, "Cho", "Cho"], [12, "Chu", "Chu"], [13, "Da", "Da"], [14, "De", "De"], [15, "Do", "Do"],
        [16, "E", "E"], [17, "Fu", "Fu"], [18, "Ga", "Ga"], [19, "Ge", "Ge"], [20, "Gi", "Gi"],
        [21, "Go", "Go"], [22, "Gu", "Gu"], [23, "Gya", "Gya"], [24, "Gyo", "Gyo"], [25, "Gyu", "Gyu"],
        [26, "Ha", "Ha"], [27, "He", "He"], [28, "Hi", "Hi"], [29, "Ho", "Ho"], [30, "Hya", "Hya"],
        [31, "Hyo", "Hyo"], [32, "Hyu", "Hyu"], [33, "I", "I"], [34, "Ja", "Ja"], [35, "Ji", "Ji"],
        [36, "Ji_2", "Ji"], [37, "Jo", "Jo"], [38, "Ju", "Ju"], [39, "Ka", "Ka"], [40, "Ke", "Ke"],
        [41, "Ki", "Ki"], [42, "Ko", "Ko"], [43, "Ku", "Ku"], [44, "Kya", "Kya"], [45, "Kyo", "Kyo"],
        [46, "Kyu", "Kyu"], [47, "Ma", "Ma"], [48, "Me", "Me"], [49, "Mi", "Mi"], [50, "Mo", "Mo"],
        [51, "Mu", "Mu"], [52, "Mya", "Mya"], [53, "Myo", "Myo"], [54, "Myu", "Myu"], [55, "N", "N"],
        [56, "Na", "Na"], [57, "Ne", "Ne"], [58, "Ni", "Ni"], [59, "No", "No"], [60, "Nu", "Nu"],
        [61, "Nya", "Nya"], [62, "Nyo", "Nyo"], [63, "Nyu", "Nyu"], [64, "O", "O"], [65, "Pa", "Pa"],
        [66, "Pe", "Pe"], [67, "Pi", "Pi"], [68, "Po", "Po"], [69, "Pu", "Pu"], [70, "Pya", "Pya"],
        [71, "Pyo", "Pyo"], [72, "Pyu", "Pyu"], [73, "Ra", "Ra"], [74, "Re", "Re"], [75, "Ri", "Ri"],
        [76, "Ro", "Ro"], [77, "Ru", "Ru"], [78, "Rya", "Rya"], [79, "Ryo", "Ryo"], [80, "Ryu", "Ryu"],
        [81, "Sa", "Sa"], [82, "Se", "Se"], [83, "Sha", "Sha"], [84, "Shi", "Shi"], [85, "Sho", "Sho"],
        [86, "Shu", "Shu"], [87, "So", "So"], [88, "Su", "Su"], [89, "Ta", "Ta"], [90, "Te", "Te"],
        [91, "To", "To"], [92, "Tsu", "Tsu"], [93, "U", "U"], [94, "Wa", "Wa"], [95, "Wo", "Wo"],
        [96, "Ya", "Ya"], [97, "Yo", "Yo"], [98, "Yu", "Yu"], [99, "Za", "Za"], [100, "Ze", "Ze"],
        [101, "Zo", "Zo"], [102, "Zu", "Zu"], [103, "Zu_2", "Zu"], [104, "A", "A"], [105, "Ba", "Ba"],
        [106, "Be", "Be"], [107, "Bi", "Bi"], [108, "Bo", "Bo"], [109, "Bu", "Bu"], [110, "Bya", "Bya"],
        [111, "Byo", "Byo"], [112, "Byu", "Byu"], [113, "Cha", "Cha"], [114, "Che", "Che"], [115, "Chi", "Chi"],
        [116, "Cho", "Cho"], [117, "Chu", "Chu"], [118, "Da", "Da"], [119, "De", "De"], [120, "Di", "Di"],
        [121, "Do", "Do"], [122, "Du", "Du"], [123, "Dyu", "Dyu"], [124, "E", "E"], [125, "Fa", "Fa"],
        [126, "Fe", "Fe"], [127, "Fi", "Fi"], [128, "Fo", "Fo"], [129, "Fu", "Fu"], [130, "Ga", "Ga"],
        [131, "Ge", "Ge"], [132, "Gi", "Gi"], [133, "Go", "Go"], [134, "Gu", "Gu"], [135, "Gya", "Gya"],
        [136, "Gyo", "Gyo"], [137, "Gyu", "Gyu"], [138, "Ha", "Ha"], [139, "He", "He"], [140, "Hi", "Hi"],
        [141, "Ho", "Ho"], [142, "Hya", "Hya"], [143, "Hyo", "Hyo"], [144, "Hyu", "Hyu"], [145, "I", "I"],
        [146, "Ja", "Ja"], [147, "Je", "Je"], [148, "Ji", "Ji"], [149, "Ji_2", "Ji"], [150, "Jo", "Jo"],
        [151, "Ju", "Ju"], [152, "Ka", "Ka"], [153, "Ke", "Ke"], [154, "Ki", "Ki"], [155, "Ko", "Ko"],
        [156, "Ku", "Ku"], [157, "Kya", "Kya"], [158, "Kyo", "Kyo"], [159, "Kyu", "Kyu"], [160, "Ma", "Ma"],
        [161, "Me", "Me"], [162, "Mi", "Mi"], [163, "Mo", "Mo"], [164, "Mu", "Mu"], [165, "Mya", "Mya"],
        [166, "Myo", "Myo"], [167, "Myu", "Myu"], [168, "N", "N"], [169, "Na", "Na"], [170, "Ne", "Ne"],
        [171, "Ni", "Ni"], [172, "No", "No"], [173, "Nu", "Nu"], [174, "Nya", "Nya"], [175, "Nyo", "Nyo"],
        [176, "Nyu", "Nyu"], [177, "O", "O"], [178, "Pa", "Pa"], [179, "Pe", "Pe"], [180, "Pi", "Pi"],
        [181, "Po", "Po"], [182, "Pu", "Pu"], [183, "Pya", "Pya"], [184, "Pyo", "Pyo"], [185, "Pyu", "Pyu"],
        [186, "Ra", "Ra"], [187, "Re", "Re"], [188, "Ri", "Ri"], [189, "Ro", "Ro"], [190, "Ru", "Ru"],
        [191, "Rya", "Rya"], [192, "Ryo", "Ryo"], [193, "Ryu", "Ryu"], [194, "Sa", "Sa"], [195, "Se", "Se"],
        [196, "Sha", "Sha"], [197, "She", "She"], [198, "Shi", "Shi"], [199, "Sho", "Sho"], [200, "Shu", "Shu"],
        [201, "So", "So"], [202, "Su", "Su"], [203, "Ta", "Ta"], [204, "Te", "Te"], [205, "Ti", "Ti"],
        [206, "To", "To"], [207, "Tsa", "Tsa"], [208, "Tse", "Tse"], [209, "Tso", "Tso"], [210, "Tsu", "Tsu"],
        [211, "Tu", "Tu"], [212, "U", "U"], [213, "Wa", "Wa"], [214, "We", "We"], [215, "Wi", "Wi"],
        [216, "Wo", "Wo"], [217, "Wo_2", "Wo"], [218, "Ya", "Ya"], [219, "Yo", "Yo"], [220, "Yu", "Yu"],
        [221, "Za", "Za"], [222, "Ze", "Ze"], [223, "Zo", "Zo"], [224, "Zu", "Zu"], [225, "Zu_2", "Zu"]
    ]
    return letter_labels

def inputData(data, k_dir, h_dir):
    letter_labels = dataLabeling()
    for label_num, label in enumerate(letter_labels):
        if label[0] >= 104:
            for file in os.listdir(os.path.join(k_dir, "Katakana_"+label[1])):
                data.append(['{}/{}/{}'.format(k_dir,
                                                "Katakana_"+label[1],
                                                file),
                                label_num, label[1]])
        else :
            for file in os.listdir(os.path.join(h_dir, "Hiragana_"+label[1])):
                data.append(['{}/{}/{}'.format(h_dir,
                                                "Hiragana_"+label[1],
                                                file),
                                label_num, label[1]])
    return data


