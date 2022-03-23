import cv2
import os
import random
import numpy as np
import numpy as np
import torch
import skimage
import cv2
import itertools
from skimage import transform as stf
from PIL import Image
from math import floor, ceil
import random


def noisy(image, noise_typ='gauss', factor=3):
    if noise_typ == "gauss":
        row, col, dim = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, dim)) * factor
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == 'poisson':
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals)) * factor
        noisy = np.uint8(np.random.poisson(image * vals) / float(vals))
        return noisy

    elif noise_typ == 'speckle':
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + np.uint8(image * gauss)
        return noisy


def lineup(im):
    vert_lines = [random.randint(0, im.shape[1] - 1) for x in range(15)]
    horiz_lines = [100, 120, 80, 60]
    choices = [0, 1, 2, 3, 4]
    space = random.choice(horiz_lines)
    orig = space
    for x_cord in vert_lines:
        for extra in range(random.choice(choices)):
            alt_coord = x_cord - extra
            if alt_coord < 0:
                pass
            else:
                im[:, alt_coord].fill(random.randint(0, 250))
    while space < im.shape[0]:
        for extra in range(random.randint(2, 4)):
            alt_coord = space - extra
            if alt_coord < 0:
                pass
            else:
                im[alt_coord, :].fill(random.randint(0, 250))
        space += orig
    return im


class Distort3():

    def __init__(self, probability, grid_width, grid_height, magnitudeX, magnitudeY, Isize, min_h_sep, min_v_sep):

        self.probability = probability
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.xmagnitude = abs(magnitudeX)
        self.ymagnitude = abs(magnitudeY)
        self.randomise_magnitude = True

        w, h = Isize

        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for x in range(horizontal_tiles)] for y in range(vertical_tiles)]
        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

                sm_h = min(self.xmagnitude, width_of_square - (min_h_sep + shift[vertical_tile][horizontal_tile - 1][
                    0])) if horizontal_tile > 0 else self.xmagnitude
                sm_v = min(self.ymagnitude, height_of_square - (min_v_sep + shift[vertical_tile - 1][horizontal_tile][
                    1])) if vertical_tile > 0 else self.ymagnitude

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))

        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)
        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)
        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for id, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[id][0]
            dy = shift[id][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                           x2, y2,
                           x3 + dx, y3 + dy,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                           x2 + dx, y2 + dy,
                           x3, y3,
                           x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                           x2, y2,
                           x3, y3,
                           x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                           x2, y2,
                           x3, y3,
                           x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        self.generated_mesh = generated_mesh

    def perform_operation(self, image):
        return image.transform(image.size, Image.MESH, self.generated_mesh, resample=Image.BICUBIC)


def aug_ED2(imgs, w, h, n_ch, tst=False):
    d = Distort3(1.0, 5, 5, 2, 5, [w, h], 1, 1)

    for i in range(len(imgs)):
        res = d.perform_operation(Image.fromarray(np.squeeze((imgs[i] * 255).astype(np.uint8))))
        imgs[i] = np.reshape(res, [h, w, n_ch])

    return np.squeeze(imgs)


def wavy(image_data):
    image_data = image_data / 255
    image_data = skimage.img_as_float32(image_data)
    if image_data.ndim < 3:
        image_data = np.expand_dims(image_data, axis=-1)

    images = image_data[None, ...]
    sh = images.shape

    image = aug_ED2(images[0][None, ...], sh[2], sh[1], sh[3], tst=False)
    if image.ndim < 3:
        image = np.expand_dims(image, axis=-1)
    return image


def slant(im, selec=random.choice([-1, 1]), ratio=random.choice([30, 40])):
    roll_tally = 0
    roll_amnt = 0
    im = np.uint8(im)
    padd = im.shape[1] // ratio
    padd = np.zeros((padd, im.shape[1], im.shape[2]), np.uint8)
    if selec == 1:
        im = np.concatenate((im, padd), axis=0)
    else:
        im = np.concatenate((padd, im), axis=0)
    #  padd += 255
    np.pad(im, 0, )
    for x in range(im.shape[1]):
        im[:, x] = np.roll(im[:, x], selec * roll_amnt, axis=0)
        roll_tally += 1
        roll_amnt = roll_tally // ratio
    ### new part here!
    roll_tally = 0
    roll_amnt = 0
    padd = im.shape[0] // ratio
    # padd = np.zeros((im.shape[0], padd, im.shape[2]), np.uint8)
    # padd.fill(random.randint(0, 255))
    padd = np.zeros((im.shape[0], padd, im.shape[2]), np.uint8)
    padd.fill(random.randint(0, 255))
    if selec == -1:
        im = np.concatenate((im, padd), axis=1)
    else:
        im = np.concatenate((padd, im), axis=1)
    # padd += 255
    np.pad(im, 0, )
    for x in range(im.shape[0]):
        im[x, :] = np.roll(im[x, :], -selec * roll_amnt, axis=0)
        roll_tally += 1
        roll_amnt = roll_tally // ratio
    return im


def slant_dim(img):
    img1 = slant(img, selec=1)
    img1 = slant(img1, selec=-1)
    return img1


def margin(im):
    boundary1 = random.randint(0, 255)
    boundary2 = random.randint(boundary1, 256)
    boundary2 = boundary1
    if boundary1 == boundary2:
        boundary1 -= 1
    if random.choice([True, False]):
        border_top = np.uint8(
            np.random.randint(boundary1, boundary2, (random.randint(25, 250), im.shape[1], im.shape[2])))
        im = np.concatenate([border_top, im])
    if random.choice([True, False]):
        border_side = np.uint8(
            np.random.randint(boundary1, boundary2, (im.shape[0], random.randint(25, 250), im.shape[2])))
        im = np.concatenate([border_side, im], axis=1)
    if random.choice([True, False]):
        border_side = np.uint8(
            np.random.randint(boundary1, boundary2, (im.shape[0], random.randint(25, 250), im.shape[2])))
        im = np.concatenate([im, border_side], axis=1)
    if random.choice([True, False]):
        border_top = np.uint8(
            np.random.randint(boundary1, boundary2, (random.randint(25, 250), im.shape[1], im.shape[2])))
        im = np.concatenate([im, border_top])
    im = np.uint8(im)
    return im


def lineup(im1):
    im = im1.copy()
    line_spacing = im.shape[0] // random.randint(20, 31)
    vert_line_cap = im.shape[1] // 4
    vert_line_start = im.shape[1] // 10
    imprint = np.zeros(im.shape)
    imprint.fill(255)
    loc = line_spacing
    choice = random.randint(vert_line_start, vert_line_cap)
    to_conc = np.random.randint(240, 255, (im.shape[0], choice), np.uint8)
    im = np.concatenate([to_conc, im], axis=1)
    im[:, choice - random.randint(2, 4):choice].fill(random.randint(0, 20))
    vert_choice = choice
    while loc < im.shape[0]:
        width = random.randint(2, 4)
        im[loc - width:loc, :].fill(0)
        loc += line_spacing
    choice = im.shape[0] // random.randint(6, 8)
    to_conc = np.random.randint(240, 255, (choice, im.shape[1]), np.uint8)
    im = np.concatenate([to_conc, im], axis=0)
    im[choice - random.randint(2, 4):choice, :].fill(random.randint(0, 20))
    if vert_choice != None:
        im[:, vert_choice - random.randint(2, 4):vert_choice].fill(random.randint(0, 20))
    image2 = cv2.resize(im, dsize=(800, 1080), interpolation=cv2.INTER_LINEAR)
    return image2


def size_up(im1):
    im = im1.copy()
    pref_size_ratio = 800 / 1080
    cur_ratio = im.shape[0] / im.shape[1]
    x = pref_size_ratio / cur_ratio
    amnt_conc = int(x * im.shape[0])
    arr = np.random.randint(240, 255, (amnt_conc, im.shape[1]), np.uint8)
    to_return = np.concatenate([im, arr], axis=0)
    return to_return

    # image2 = cv2.resize(image, dsize=(image.shape[1] // 3, image.shape[0] // 3), interpolation=cv2.INTER_CUBIC)


def to_mean(image):
    choice = False  # random.choice([True, False])
    avg = image.mean()
    if not choice:
        avg = avg - random.randint(45, 200)
        tick = False
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            val = (avg + image[i][j]) // 2 + random.randint(-10, 10)
            image[i][j] = min(max(val, 0), 255)
    return image


if __name__ == '__main__':
    dir = '/Users/petergeraghty/ocr_experiments/iam_data/pargs/'
    for im in os.listdir(dir):
        if im[-3:] == 'png':
            im = cv2.imread(dir+im, 0)
            imr = size_up(im)
            imr = lineup(imr)
            imr = to_mean(imr)
            dst = cv2.GaussianBlur(imr, (5, 5), cv2.BORDER_DEFAULT)
            imr = np.uint8(wavy(dst))
            cv2.imshow('image', imr)
            cv2.waitKey(0)
