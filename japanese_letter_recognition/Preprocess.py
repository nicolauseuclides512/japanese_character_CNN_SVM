import cv2
import os
import glob
import japanese_letter_recognition.FileNameList as listfname


def bigPreprocess(main_directory, directoryname, files):
    data = []
    dsize = (42, 42)
    for f1 in files:
        img = cv2.imread(f1)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        new_image = cv2.resize(blackAndWhiteImage, dsize)
        data.append(new_image)
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
