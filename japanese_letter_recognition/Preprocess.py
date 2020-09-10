import cv2
import os
import glob
import numpy as np
import csv
import japanese_letter_recognition.FileNameList as listfname
from sklearn.model_selection import train_test_split
from japanese_letter_recognition import NeuralNetwork as convn
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D


def bigPreprocess(main_directory, directoryname, img_dir, files, datas, labels):
    data = []
    label = []
    kernel = np.ones((3, 3), np.uint8)
    kernel_morph = np.ones((2, 2), np.uint8)
    if "Hiragana" in img_dir:
        # os.chdir(main_directory + 'blackwhite_image/Hiragana')
        label = listfname.hiragana_label
    elif "Katakana" in img_dir:
        # os.chdir(main_directory + 'blackwhite_image/Katakana')
        label = listfname.katakana_label
    os.chdir(directoryname)
    for f1 in files:
        img = cv2.imread(f1)
        img_resize = cv2.resize(img, (168, 168))
        grayImage = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 1, cv2.THRESH_BINARY)
        img_erotion = cv2.erode(blackAndWhiteImage, kernel, iterations=2)
        img_morph_open = cv2.morphologyEx(img_erotion, cv2.MORPH_OPEN, kernel_morph, iterations=3)
        img_morph_close = cv2.morphologyEx(img_morph_open, cv2.MORPH_CLOSE, kernel_morph, iterations=3)
        img_dilation = cv2.dilate(img_morph_close, kernel, iterations=1)
        (x, y) = img_dilation.shape
        # img_dilation = list(np.asarray(img_dilation))

        for lname in label:
            if lname[1] == img_dir.partition("_")[2]:
                data.append([img_dilation, lname[0]])
    for f2 in range(0, len(data)):
        datas.append(data[f2][0])
        labels.append(data[f2][1])
    # data_train, data_test, label_train, label_test = train_test_split(datas, labels, test_size=0.33)
    # data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=0.25,
    #                                                                 random_state=1)  # 0.25 x 0.8 = 0.2
    # data_trains.append(data_train)
    # data_tests.append(data_test)
    # data_vals.append(data_val)
    # label_trains.append(label_train)
    # label_tests.append(label_test)
    # label_vals.append(label_val)
    print(img_dir + ' done')


# def checkDirectoryForSave(directoryname, data):
#     if os.path.exists(directoryname):
#         os.chdir(directoryname)
#         saveImage(data, directoryname)
#     else:
#         os.mkdir(directoryname)
#         os.chdir(directoryname)
#         saveImage(data, directoryname)


# def saveImage(data, filename):
#     for f2 in range(0, len(data)):
#         if os.path.exists('{}'.format(filename + '_' + format(f2 + 1) + '.jpg', data[f2])) == 0:
#             cv2.imwrite(filename + '_' + format(f2 + 1) + '.jpg', data[f2])


def hiraganaPreprocess(main_directory, directory_name, datas, labels):
    for fname in listfname.hiragana_name:
        os.chdir(directory_name)
        img_dir = format(fname)
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)
        bigPreprocess(main_directory, directory_name, img_dir, files, datas, labels)


def katakanaPreprocess(main_directory, directory_name, datas, labels):
    for fname in listfname.katakana_name:
        os.chdir(directory_name)
        img_dir = format(fname)
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)
        bigPreprocess(main_directory, directory_name, img_dir, files, datas, labels)


def saveFile(data, filename, main_directory):
    os.chdir(main_directory + 'japanese_letter_recognition')
    if "Hiragana" in filename:
        os.chdir(main_directory + 'blackwhite_image/Hiragana')
        # checkDirectoryForSave(filename, data)
    elif "Katakana" in filename:
        os.chdir(main_directory + 'blackwhite_image/Katakana')
        # checkDirectoryForSave(filename, data)
    # if os.path.exists('Kana_Letter.csv') == 0:
    #     with open('Kana_Letter.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         writer.writerow(['data', 'label'])
    #         writer.writerows(save_data)
    # else:
    #     with open('Kana_Letter.csv', 'a+', newline='') as csvfile:
    #         writer = csv.writer(csvfile, delimiter=',')
    #         writer.writerows(save_data)
