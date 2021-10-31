
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
from tflite_runtime.interpreter import Interpreter
import scipy.ndimage


def getFinalOutput(numlist, interpreter, characters):
    return ''.join(decode(characters, interpreter.get_tensor(x["index"])) for x in numlist).replace("&", "")

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=1)
    return ''.join([characters[x] for x in y])

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

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with open(args.output, 'w') as output_file:
        output_file.write("vnagaraj/n")
        interpreter = Interpreter("model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dirlist = os.listdir(args.captcha_dir)
        list.sort(dirlist)
        for x in dirlist:
                # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = removeNoise(rgb_data)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], [image])

            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            output_file.write(x + "," + getFinalOutput(output_details, interpreter, captcha_symbols) + "\n")

            print('Classified ' + x)

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

if __name__ == '__main__':
    main()
