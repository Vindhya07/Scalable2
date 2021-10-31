#!/usr/bin/env python3

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import scipy.ndimage
import math


def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:, 0]
    y_dict = dict(enumerate(characters))
    return ''.join([y_dict.get(x, '') for x in y]).replace("&", "")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    dirlist = os.listdir(args.captcha_dir)
    list.sort(dirlist)

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            output_file.write("vnagaraj\n")
            json_file = open(args.model_name + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            h5File = args.model_name + '.h5'
            model.load_weights(h5File)
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for x in dirlist:
                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                # rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                # image = numpy.array(rgb_data) / 255.0
                result = removeNoise(raw_data)
                # (c, h, w) = result.shape
                result = result.reshape([-1, 64, 128, 1])
                prediction = model.predict(result)
                output_file.write(x + "," + decode(captcha_symbols, prediction) + "\n")

                print('Classified' + x)

def removeNoise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = ~gray
    img = cv2.erode(img, numpy.ones((2, 2), numpy.uint8), iterations=1)
    img = ~img  # black letters, white background
    img = scipy.ndimage.median_filter(img, (5, 1))
    img = scipy.ndimage.median_filter(img, (1, 1))
    thresh = ~cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20:
            cv2.drawContours(thresh, [c], -1, 0, -1)

    result = 255 - thresh
    return cv2.GaussianBlur(result, (3, 3), 0)


def getX(coord):
    list = []
    for a in coord:
        list.append(a[0][0])
    return list

def getY(coord):
    list = []
    for a in coord:
        list.append(a[0][1])
    return list


def preprocessImage(image):
    output = removeNoise(image)
    contours, _ = cv2.findContours(255 - output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minList = []
    maxList = []
    imgDict = {}
    minyList = []
    maxyList = []
    a = []
    count = 0
    area = 0
    for cont in contours:
        isUpdated = 0
        xlist = getX(cont)
        ylist = getY(cont)

        min_x = min(xlist)
        max_x = max(xlist)
        min_y = min(ylist)
        max_y = max(ylist)

        for i in range(len(minList)):
            if min_x > minList[i] and min_x < maxList[i]:
                imgDict.update({minList[i]: output[:, minList[i]: max_x]})
                maxList[i] = max_x
                isUpdated = 1
                break;
        if not isUpdated:
            minList.append(min_x)
            maxList.append(max_x)
            minyList.append(min_y)
            maxyList.append(max_y)
            imgDict.update({min_x: output[:, min_x: max_x]})
        area += cv2.contourArea(cont)


    for k in sorted(imgDict.keys()):
        a.append(imgDict[k])
    if len(minList) != 0:
        output_without_space = numpy.concatenate(a, axis=1)

        (w, h) = numpy.shape(output_without_space)

        space_array = numpy.ones((w, 128 - h), dtype=numpy.uint8) * 255

        result = numpy.concatenate((output_without_space, space_array), axis=1)
    else:
        result = output
    return result


if __name__ == '__main__':
    main()
