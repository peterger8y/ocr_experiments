import difflib
import logging
import math
import string
import random
import cv2
import os

import numpy as np
import mxnet as mx
from tqdm import tqdm

from paragraph_segmentation_dcnn import SegmentationNetwork, paragraph_segmentation_transform
from word_and_line_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

from utils.iam_dataset import IAMDataset, crop_handwriting_page
from utils.sclite_helper import ScliteHelper
from utils.word_to_line import sort_bbs_line_by_line, crop_line_images
from utils.draw_box_on_image import draw_box_on_image

# Setup
logging.basicConfig(level=logging.DEBUG)
random.seed(123)
np.random.seed(123)
mx.random.seed(123)

# Input sizes
form_size = (1120, 800)
segmented_paragraph_size = (800, 800)
line_image_size = (60, 800)

# Parameters
min_c = 0.01
overlap_thres = 0.001
topk = 400
rnn_hidden_states = 512
rnn_layers = 2
max_seq_len = 160


word_segmentation_model = "/models/word_segmentation2.params"


def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]


denoise_func = get_arg_max

if __name__ == '__main__':

    # Compute context
    ctx = mx.gpu(0)

    # Models

    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters(word_segmentation_model, ctx)

    logging.info("models loaded.")

    # Data
    logging.info("loading data...")
    # test_ds = IAMDataset("form_original", credentials=('petergeraghty57@gmail.com', 'bewros-zorNaz-5cicjo'), train=False)
    logging.info("data loaded.")

    dir = '/content/drive/MyDrive/dataset/'
    alpha = 1
    for i, path in enumerate(os.listdir(dir)):
        if i == 20:
            print('complete, check your folder!!')
            break
        image = cv2.imread(dir + path, 0)
        resized_image = paragraph_segmentation_transform(image, image_size=form_size)
        word_bb = predict_bounding_boxes(word_segmentation_net, image, min_c, overlap_thres, topk, ctx)
        for box in word_bb:
            xmin = int(box[0] * image.shape[1])
            ymin = int(box[1] * image.shape[0])
            xmax = int(xmin + box[2] * image.shape[1])
            ymax = int(ymin + box[3] * image.shape[0])
            cv2.line(image, (xmin, ymin), (xmax, ymin), (0, 0, 255), 2)
            cv2.line(image, (xmax, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.line(image, (xmax, ymax), (xmin, ymax), (0, 0, 255), 2)
            cv2.line(image, (xmin, ymax), (xmin, ymin), (0, 0, 255), 2)
        cv2.imshow('image', image)
        alpha += 1