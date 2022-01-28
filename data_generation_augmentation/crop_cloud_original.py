# Function to produce a json file for web to display a Plotly line graph that
# maps the history of a specific student's submission scores

# Imports
import os
import base64
from math import sqrt
import random
import json
import zipfile
import numpy as np
import pandas as pd
import cv2


# resize so the longest side is max_length
def load_image(filename, max_length=None):
    image = cv2.imread(filename)
    if max_length:
        original_length = max(image.shape[:2])
        scale_factor = max_length / original_length
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    return image


# Convert the image into black and white, separating the text from the background
def make_monochrome(image, blur=1, block_size=31, c=13):
    # blur must be an odd number >= 1
    # block_size is for smoothing out a varying exposure. too small etches out text. must be an odd number > 1
    # c is for denoising. too small and you have noise. too big erodes text.

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    image = cv2.adaptiveThreshold(
        src=image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=c)
    if blur > 1:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image


# Resizes the cropped words for a given canvas area and density
def scale_clip(boxes, canvas):
    scaled_clips = []
    standard = 1 / 150 * (canvas.shape[0] * canvas.shape[1])
    for box in boxes:
        random_sizing = 3 * random.uniform(.125, 1.)
        box_area = random_sizing * (box.shape[0] * box.shape[1])
        ratio = np.sqrt(standard / box_area)
        second = int(ratio * box.shape[0])
        first = int(ratio * box.shape[1])
        size = (first, second)
        if ratio < 1:
            resize_algo = cv2.INTER_AREA  # recommended for shrinking
        else:
            resize_algo = cv2.INTER_CUBIC  # recommended for enlarging
        image = cv2.resize(box, dsize=size, interpolation=resize_algo)
        scaled_clips.append(image)
    return scaled_clips


# Collate all the requested pages into one words table, including the file path, date, and cropped words
def get_user_words():
    queue = []
    for x in os.listdir('crop_cloud_data_gen/'):
        queue.append(x)
    random.shuffle(queue)
    return queue


def load_snippets(words):
    dir = 'crop_cloud_data_gen/'
    images = []
    for x in words:
        im = cv2.imread(dir + x)
        images.append(im)
    return images


# Picks a random horizontal location for a cropped word
# This is the heart of arranging the words chronologically
def pick_x(canvas_width, word_width, date_number=None, total_dates=None):
    if total_dates:  # this is untested and may need to be debugged
        # for now, this divides the space into even fractions
        # I had wanted to use arc cosine waves to give a fuzzy distribution, but never finished the function bending
        x_float = np.random.uniform(
            low=date_number / total_dates,
            high=(date_number + 1) / total_dates,
        )
    else:
        x_float = np.random.uniform()
    available_room = canvas_width - word_width
    return int(x_float * available_room)


# Picks a random vertical location for a cropped word
# This uses a triangular distribution which biases the words towards the midline, where your eyes will start
def pick_y(canvas_height, word_height):
    y_float = np.random.triangular(0, 0.5, 1)
    available_room = canvas_height - word_height
    return int(y_float * available_room)


# Constructs and renders a crop cloud. Returns an image
def make_page(canvas, boxes, words):
    # propose a location for the word
    # does this location collide with anything already placed?
    # if not, then place the word
    # OpenCV uses [y:x] coordinates for images

    occupied = np.zeros(shape=(canvas.shape[:2]), dtype=bool)
    placed = 0
    collisions = 0
    total = len(boxes)
    word_area = 0
    for i, row in enumerate(boxes):
        image = row
        max_attempts = 20
        failed_attempts = 0
        while failed_attempts < max_attempts:
            # pick a horizontal position
            x1 = pick_x(
                canvas_width=canvas.shape[1],
                word_width=image.shape[1],
                date_number=0,
                total_dates=1,
            )
            x2 = x1 + image.shape[1]

            # pick a vertical position
            y1 = pick_y(canvas_height=canvas.shape[0], word_height=image.shape[0])
            y2 = y1 + image.shape[0]

            R, G, B = cv2.split(image)  # split the image into channels
            mask = 255 - make_monochrome(image)
            color = cv2.merge((B, G, R))
            mask = np.atleast_3d(mask) / 255

            mask_bool = mask.reshape(mask.shape[:2]).astype('bool')
            mask_bool = np.ones((image.shape[0], image.shape[1])).astype(bool)
            intersection = np.logical_and(mask_bool, occupied[y1:y2, x1:x2])

            if intersection.sum() > 0:
                # reject this placement
                failed_attempts += 1
                collisions += 1
                continue
            else:
                # place the word
                canvas[y1:y2, x1:x2] = (mask * color + (1 - mask) * canvas[y1:y2, x1:x2])
                occupied[y1:y2, x1:x2] = np.logical_or(mask_bool, occupied[y1:y2, x1:x2])
                word_area += (x2 - x1) * (y2 - y1)
                placed += 1
                break

    return canvas


def get_crop_cloud(canvas_width=1024):
    """
    Renders and returns a whole crop cloud for a user's submissions over a given date range

    Input:
        `user_id` str - a string containing the username
        `date_range` List[str] - a list of two dates in the format of YYYY-MM-DD
        `complexity_metric` str - how to calculate the complexity of words (from 'len', 'syl', 'len_count', 'syl_count')
        `image_format` str - the format of the cropped word images (from '.png', '.webp', or anything OpenCV supports)
        `width` int - the width of the crop cloud in pixels
        `density` float - the bounding box area of the cropped words divided by the canvas area
        `max_words` int - the max number of words to include in the cloud

    Output:
        json(image_base64) - a rendered crop cloud as an image
    """

    user_words = get_user_words()
    num_words = random.randint(0, 150)
    words_to_use = user_words[:num_words]
    snippets = load_snippets(words_to_use)
    canvas = np.zeros((731, 1024, 3), np.uint8)
    canvas.fill(255)
    print(words_to_use)
    snippets = scale_clip(snippets, canvas)

    page = make_page(canvas, snippets, words_to_use)

    return page


if __name__ == "__main__":
    for i in range(10):
        alpha = get_crop_cloud()
        cv2.imshow('image', alpha)
        cv2.waitKey(0)

    # for x in os.listdir('crop_cloud_data_gen/'):
    #     if len(x.split('_')[0]) < 4:
    #         if random.choice([True, False]):
    #             print(x.split('_')[0])
    #             os.remove('crop_cloud_data_gen/' + x)
