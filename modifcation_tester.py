import data_modification
import random
import os
import numpy as np
import cv2

def blur(img, choice = random.randint(6, 20)):
    img = np.uint8(img)
    img = cv2.blur(img, (choice, choice))
    print(img)
    print(choice)
    return img


if __name__ == '__main__':
    dir = 'iam_data/pargs/'
    for im in os.listdir(dir):
        if im[-3:] == 'png':
            image = cv2.imread(dir+im, 0)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            integer = random.randint(10, 25)
            ksize = (integer, integer)
            image = blur(image)
            cv2.imshow('image post blur', image)
            cv2.waitKey(0)