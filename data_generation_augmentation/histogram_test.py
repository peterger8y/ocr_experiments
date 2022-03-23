import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy.stats
import os
import random

def det_bot_word(word1):
    image = cv2.adaptiveThreshold(
        src=np.uint8(word1),
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=51)
    image = (1 - (image / 255))
    print(image)
    cumulative_vals = [np.sum(image[i, :]) for i in range(image.shape[0])]
    print(cumulative_vals)
    total = 0
    tally = 0
    new_arr = []
    mode = None
    for i, x in enumerate(cumulative_vals):
        total += x*i
        tally += x
        for j in range(int(x)):
            new_arr.append(i)
        if mode is None or x > mode:
            mode = i
    tar = int(total/tally)
    std = int(np.std(new_arr))
    std = int(std * random.uniform(.8, 1.15))
    print(random.uniform(.8, 1.15))
    word1[min(word1.shape[0]-1, tar+std), :].fill(0)
    return word1

if __name__ == '__main__':
    # image = cv2.imread('/Users/petergeraghty/ocr_experiments/crop_cloud_data_gen/afford_5523.png', 0)
    dir = '/Users/petergeraghty/ocr_experiments/crop_cloud_data_gen/'
    for x in os.listdir(dir):
        image = cv2.imread(dir+x, 0)
        modified_word = det_bot_word(image)
        cv2.imshow('image', modified_word)
        cv2.waitKey(0)
    # Now we can save it to a numpy array.