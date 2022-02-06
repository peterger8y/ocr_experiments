import numpy as np
import cv2
import os

def test():
    boxes_txt = open('/Users/petergeraghty/ocr_experiments/bounding_boxes/bb_outputs/Photo_3101.txt', 'r').read()
    image = cv2.imread('/Users/petergeraghty/ocr_experiments/bounding_boxes/bb_outputs/Photo_3101.jpg', 0)
    blank = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    for box in boxes_txt.split('\n'):
        if len(box) == 0:
            continue
        box = box.split(',')
        for i, x in enumerate(box):
            box[i] = int(box[i])
        coord1 = (box[0], box[1])
        coord2 = (box[2], box[3])
        coord3 = (box[4], box[5])
        coord4 = (box[6], box[7])
        cv2.line(blank, coord1, coord2, (0, 0, 255), 2)
        cv2.line(blank, coord2, coord3, (0, 0, 255), 2)
        cv2.line(blank, coord3, coord4, (0, 0, 255), 2)
        cv2.line(blank, coord4, coord1, (0, 0, 255), 2)
    cv2.imshow('blank', blank)
    cv2.waitKey(0)


if __name__ == '__main__':
    test()