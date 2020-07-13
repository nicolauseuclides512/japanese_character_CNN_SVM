import pathlib

import cv2
import os
import glob
import japanese_letter_recognition.fileNameList
import subprocess

# def saveImage(data):
#     for f2 in range(0, len(data)):
#         if os.path.exists('{}'.format(format(fname) + '_' + format(f2 + 1) + '.jpg', data[f2])) == 0:
#             cv2.imwrite(format(fname) + '_' + format(f2 + 1) + '.jpg', data[f2])
for fname in japanese_letter_recognition.fileNameList.hiragana_name:
    os.chdir('/home/nicolaus/PycharmProjects/thesis_program/data_image/Hiragana')
    img_dir = format(fname)
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    dsize = (42, 42)

    for f1 in files:
        img = cv2.imread(f1)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        new_image = cv2.resize(blackAndWhiteImage, dsize)
        data.append(new_image)

    os.chdir('/home/nicolaus/PycharmProjects/thesis_program/blackwhite_image/Hiragana')
    if os.path.exists('{}'.format(format(fname))):
        os.chdir(format(fname))
        for f2 in range(0, len(data)):
            if os.path.exists('{}'.format(format(fname) + '_' + format(f2 + 1) + '.jpg', data[f2])) == 0:
                cv2.imwrite(format(fname) + '_' + format(f2 + 1) + '.jpg', data[f2])
    else:
        os.mkdir(format(fname))
        os.chdir(format(fname))
        for f2 in range(0, len(data)):
            if os.path.exists('{}'.format(format(fname) + '_' + format(f2 + 1) + '.jpg', data[f2])) == 0:
                cv2.imwrite(format(fname) + '_' + format(f2 + 1) + '.jpg', data[f2])
    print(format(fname) + ' done')
