import numpy as np
import cv2
import os
import random


def concatenator_p2():
    dir = 'iam_data/pargs/'
    tar_dir = 'iam_data/bigger_text_set/'
    i = 0
    j = 0
    alpha = 0
    for path in os.listdir(dir):
        if path[-3:] == 'png':
            im_curr = cv2.imread(dir + path, 0)
            if i == 0:
                boundary1 = random.randint(0, 256)
                boundary2 = random.randint(boundary1, 255)
                if boundary1 == boundary2:
                    boundary1 -= 1
                padd = np.random.randint(boundary1, boundary2, (im_curr.shape), np.uint8)
                if random.choice([True, False]):
                    im_curr = np.concatenate([im_curr, padd])
                else:
                    im_curr = np.concatenate([padd, im_curr])
                cv2.imwrite(tar_dir + str(alpha) + '.png', im_curr)
                with open(dir + path[:-3] + 'txt', 'r') as filed:
                    text = filed.read()
                with open(tar_dir + str(alpha) + '.txt', 'x') as filed:
                    filed.write(text)
                alpha += 1
            elif i == 1:
                im_prev = im_curr
                with open(dir + path[:-3] + 'txt', 'r') as filed:
                    string_along = filed.read()
            elif i == 2:
                im_second_prev = im_curr
                with open(dir + path[:-3] + 'txt', 'r') as filed:
                    text = filed.read()
                    string_along += ' ' + text
            elif i == 3:
                im_prev = cv2.resize(im_prev, (im_curr.shape[1], im_curr.shape[0]), interpolation=cv2.INTER_AREA)
                im_second_prev = cv2.resize(im_second_prev, (im_curr.shape[1], im_curr.shape[0]), interpolation=cv2.INTER_AREA)
                print(im_prev.shape, im_second_prev.shape, im_curr.shape)
                im_newer = np.concatenate([im_prev, im_second_prev])
                im_curr = np.concatenate([im_curr, im_newer])
                cv2.imwrite(tar_dir + str(alpha) + '.png', im_curr)
                with open(dir + path[:-3] + 'txt', 'r') as filed:
                    text = filed.read()
                    string_along += ' ' + text
                with open(tar_dir + str(alpha) + '.txt', 'x') as filed:
                    filed.write(string_along)
                alpha += 1
            elif i == 4:
                cv2.imwrite(tar_dir + str(alpha) + '.png', im_curr)
                with open(dir + path[:-3] + 'txt', 'r') as filed:
                    text = filed.read()
                with open(tar_dir + str(alpha) + '.txt', 'x') as filed:
                    filed.write(text)
            i += 1
            i = i % 4
            cv2.imshow('image', im_curr)
            cv2.waitKey(0)


def concatenator():
    dir = 'iam_data/pargs/'
    for path in os.listdir(dir):
        if path[-3:] == 'png':
            im = cv2.imread(dir + path, 0)
            if random.choice([True, False]):
                border_top = np.random.randint(0, 255, (random.randint(25, 100), im.shape[1]))
                im = np.concatenate([border_top, im])
            if random.choice([True, False]):
                border_side = np.random.randint(0, 255, (im.shape[0], random.randint(25, 100)))
                im = np.concatenate([border_side, im], axis=1)
            if random.choice([True, False]):
                border_side = np.random.randint(0, 255, (im.shape[0], random.randint(25, 100)))
                im = np.concatenate([im, border_side], axis=1)
            if random.choice([True, False]):
                border_top = np.random.randint(0, 255, (random.randint(25, 100), im.shape[1]))
                im = np.concatenate([im, border_top])
            im = np.uint8(im)
            return im
            cv2.imshow('image', np.uint8(im))
            cv2.waitKey(0)


if __name__ == '__main__':
    concatenator_p2()
