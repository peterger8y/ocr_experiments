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
import noise


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


def scale_clip2(boxes, canvas, words):
    scaled_clips = []
    for i, box in enumerate(boxes):
        length = len(words[i].split('_')[0])
        box_area = (box.shape[0] * box.shape[1])
        standard = ((length * .2) + .5) / 800 * (canvas.shape[0] * canvas.shape[1]) * random.uniform(.8, 1.2)
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
    num_words = random.randint(0, 250)
    words_to_use = user_words[:num_words]
    snippets = load_snippets(words_to_use)
    int1 = random.randint(240, 255)
    int2 = random.randint(int1, 255)
    canvas = np.random.randint(int1, int2 + 1, (1080, 800, 1), np.uint8)
    page, spacing, cap, left_end = noise.pageup(canvas)
    page = np.expand_dims(page, -1)
    snippets = scale_clip2(snippets, canvas, user_words)
    # snippets = scale_clip(snippets, canvas)
    # page = make_page(canvas, snippets, words_to_use, spacing, cap, left_end)
    page, text = write_lines_to_page(page, snippets, words_to_use, spacing, left_end, cap)
    page = noise.to_mean(page)
    #page = noise.wavy(page)
    # func_direct = {1: noise.noisy, 2: noise.lineup, 3: noise.uniform_lineup, 4: noise.slant}
    # choices = [x for x in func_direct.keys()]
    # num_choices = random.randint(0, len(choices))
    # dist_selec = random.sample(choices, num_choices)
    # for ind in dist_selec:
    #     page = np.uint8(func_direct[ind](page))
    # if random.choice([True, False]):
    #     page = noise.margin(page)
    page = (255 - ((255 - page) * random.uniform(.65, 1)))

    return page, spacing, cap, left_end, text


def det_bot_word(word1):
    image = cv2.adaptiveThreshold(
        src=np.uint8(word1),
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=51)
    image = (1 - (image / 255))
    cumulative_vals = [np.sum(image[i, :]) for i in range(image.shape[0])]
    total = 0
    tally = 0
    new_arr = []
    mode = None
    for i, x in enumerate(cumulative_vals):
        total += x * i
        tally += x
        for j in range(int(x)):
            new_arr.append(i)
        if mode is None or x > mode:
            mode = i
    tar = int(total / tally)
    std = int(1.1 * (np.std(new_arr)))
    #std = int(std * random.uniform(.7, 1))
    return tar+std



def write_lines_to_page(canvas, snippets, words_to_use, spacing, left_end, cap):
    num = 1
    cur_line_x = left_end
    stacked = random.choice([True, False])
    text = ''

    i = 0
    breaker = True
    page_adj = random.randint(0, 255)
    while True:
        if random.choice([True, False]):
            num += 1
        else:
            cur_line_y = int(cap + (spacing * num))
            break
    while True:
        if i >= len(snippets):
            break
        if cur_line_y >= canvas.shape[0]:
            break
        color = snippets[i]
        color = np.expand_dims(make_monochrome(color), -1)
        line_pos = det_bot_word(color)
        mask = color / 255
        if breaker:
            breaker = False
            space_pre = random.randint(0, 150)
        else:
            space_pre = random.randint(15, 75)
        y1 = cur_line_y - line_pos
        y2 = y1 + mask.shape[0]
        x1 = cur_line_x + space_pre
        x2 = x1 + mask.shape[1]
        color = color + (page_adj + color) // 2
        try:
            canvas[y1:y2, x1:x2] = (color * (1 - mask) + (mask) * canvas[y1:y2, x1:x2])
            cur_line_x += color.shape[1] + space_pre
            text += ' ' + words_to_use[i].split('_')[0]
            i += 1
        except:
            if not stacked:
                num += random.randint(1, 3)
                cur_line_y = cap + int(spacing * num)
            else:
                num += 1
                cur_line_y = cap + int(num * spacing)
            cur_line_x = left_end
            breaker = True
            pass

    return canvas, text


if __name__ == "__main__":
    while True:
        alpha, spacing, cap, left_end, text = get_crop_cloud()
        cv2.imshow('image', np.uint8(alpha))
        cv2.waitKey(0)

    # for x in os.listdir('crop_cloud_data_gen/'):
    #     if len(x.split('_')[0]) < 4:
    #         if random.choice([True, False]):
    #             print(x.split('_')[0])
    #             os.remove('crop_cloud_data_gen/' + x)
