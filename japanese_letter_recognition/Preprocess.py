import cv2
import os
import glob
import numpy as np

import japanese_letter_recognition.FileNameList as listfname


def bigPreprocess(main_directory, directoryname, files):
    data = []
    kernel = np.ones((3, 3), np.uint8)
    kernel_morph = np.ones((2, 2), np.uint8)
    for f1 in files:
        img = cv2.imread(f1)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        img_resize = cv2.resize(blackAndWhiteImage, (240, 240))
        img_erotion = cv2.erode(img_resize, kernel, iterations=2)
        img_morph_open = cv2.morphologyEx(img_erotion, cv2.MORPH_OPEN, kernel_morph, iterations=3)
        img_morph_close = cv2.morphologyEx(img_morph_open, cv2.MORPH_CLOSE, kernel_morph, iterations=3)
        img_dilation = cv2.dilate(img_morph_close, kernel, iterations=1)
        img_resize = cv2.resize(img_dilation, (30, 30))
        data.append(img_resize)
    if "Hiragana" in directoryname:
        os.chdir(main_directory + 'blackwhite_image/Hiragana')
    elif "Katakana" in directoryname:
        os.chdir(main_directory + 'blackwhite_image/Katakana')
    checkDirectoryForSave(directoryname, data)
    print(directoryname + ' done')


def checkDirectoryForSave(directoryname, data):
    if os.path.exists(directoryname):
        os.chdir(directoryname)
        saveImage(data, directoryname)
    else:
        os.mkdir(directoryname)
        os.chdir(directoryname)
        saveImage(data, directoryname)


def saveImage(data, filename):
    for f2 in range(0, len(data)):
        if os.path.exists('{}'.format(filename + '_' + format(f2 + 1) + '.jpg', data[f2])) == 0:
            cv2.imwrite(filename + '_' + format(f2 + 1) + '.jpg', data[f2])


def hiraganaPreprocess(main_directory, directory_name):
    for fname in listfname.hiragana_name:
        os.chdir(directory_name)
        img_dir = format(fname)
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)
        bigPreprocess(main_directory, img_dir, files)


def katakanaPreprocess(main_directory, directory_name):
    for fname in listfname.katakana_name:
        os.chdir(directory_name)
        img_dir = format(fname)
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)
        bigPreprocess(main_directory, img_dir, files)
