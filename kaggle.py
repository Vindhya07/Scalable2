#!/usr/bin/env python3
import copy
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

width = 128
height = 64
length = 6
count = 128000
output_dir = "train"
captcha_symbols = "%{}[]()#:| 1234567890ABCDFMPQRSTUVWXYZecghjknpx"
train_dataset=""
validate_dataset=""
output_model_name="model_pro"

# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape, model_depth=5, module_size=3):
    input_tensor = keras.Input(input_shape)
    x = input_tensor
    for i, module_length in enumerate([module_size] * model_depth):
        for j in range(module_length):
            x = keras.layers.Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same',
                                    kernel_initializer='he_uniform')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.MaxPooling2D(2)(x)

    x = keras.layers.Flatten()(x)
    x = [keras.layers.Dense(captcha_num_symbols, activation='softmax', name='char_%d' % (i + 1))(x) for i in
         range(captcha_length)]
    model = keras.Model(inputs=input_tensor, outputs=x)

    return model


# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height
        # print(self.directory_name)
        file_list = os.listdir(self.directory_name)
        self.files_perm = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []
        self.count = len(file_list)

    def __len__(self):
        # print("len function")
        # print(int(numpy.floor(self.count / self.batch_size)))
        return int(numpy.floor(self.count / self.batch_size))

    def __getitem__(self, idx):
        X = numpy.zeros((self.batch_size, self.captcha_height, self.captcha_width), dtype=numpy.float32)
        y = [numpy.zeros((self.batch_size, len(self.captcha_symbols)), dtype=numpy.uint8) for i in
             range(self.captcha_length)]

        if len(self.files.keys()) == 0:
            self.files = copy.deepcopy(self.files_perm)
        a = self.files
        for i in range(self.batch_size):

            # print("Batch Size" + str(i))
            # print(self.files_perm)
            # print(self.count)
            # print("File Length" + str(len(a.keys())))
            random_image_label = random.choice(list(a.keys()))
            random_image_file = a[random_image_label]
            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(a.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            raw_data = cv2.imread(os.path.join(self.directory_name, random_image_file))
            output = preprocessImage(raw_data)

            X[i] = output

            # We have a little hack here - we save captchas as TEXT_num.png if there is more than one captcha with the text "TEXT"
            # So the real label should have the "_num" stripped out.

            random_image_label = random_image_label.split('_')[0].replace(" ", "")
            label_len = len(random_image_label)

            if label_len <= self.captcha_length:
                 random_image_label = random_image_label + ''.join([" " for u in range(self.captcha_length-label_len)])
            for j, ch in enumerate(random_image_label):
                # y[j][i, :] = 0
                # if self.captcha_symbols.find(ch) != -1:
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y


def removeNoise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    img = ~gray
    img = cv2.erode(img, numpy.ones((2, 2), numpy.uint8), iterations=1)  # weaken circle noise and line noise
    img = ~img  # black letters, white background
    img = scipy.ndimage.median_filter(img, (5, 1))  # remove line noise
    img = scipy.ndimage.median_filter(img, (1, 1))  # remove circular noise
    thresh = ~cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('thresh', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = thresh
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('opening', opening)

    # opening = cv2.erode(thresh, kernel)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 20:
            cv2.drawContours(opening, [c], -1, 0, -1)

    result = 255 - opening
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




def main():




    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "No GPU available!"
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # with tf.device('/device:GPU:0'):
    with tf.device('/device:GPU:0'):
        # with tf.device('/device:XLA_CPU:0'):
        model = create_model(length, len(captcha_symbols), (height, width, 1))



        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        # model.summary()

        training_data = ImageSequence(train_dataset, batch_size, length, captcha_symbols, width,
                                      height)
        validation_data = ImageSequence(validate_dataset, batch_size, length, captcha_symbols,
                                        width, height)

        callbacks = [keras.callbacks.EarlyStopping(patience=3),
                     # keras.callbacks.CSVLogger('log.csv'),
                     keras.callbacks.ModelCheckpoint(output_model_name + '.h5', save_best_only=False)]

        # Save the model architecture to JSON
        with open(output_model_name + ".json", "w") as json_file:
            json_file.write(model.to_json())

        try:
            model.fit_generator(generator=training_data,
                                validation_data=validation_data,
                                epochs=epochs,
                                callbacks=callbacks,
                                use_multiprocessing=False)

        except KeyboardInterrupt:
            print('KeyboardInterrupt caught, saving current weights as ' + output_model_name + '_resume.h5')
            model.save_weights(output_model_name + '_resume.h5')


if __name__ == '__main__':
    main()
