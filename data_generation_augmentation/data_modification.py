import cv2
import os
import random
import numpy as np
import skimage
import cv2
import itertools
from PIL import Image
from math import floor, ceil
import random

"""Working with the following distorition algorithms to improve performance of handwriting text recognizer"""


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
    d = Distort3(1.0, 10, 10, 5, 15, [w, h], 1, 1)

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


def slant(im, selec=random.choice([-1, 1]), ratio=random.choice([10, 20, 30, 40])):
    roll_tally = 0
    roll_amnt = 0
    im = np.uint8(im)
    padd = im.shape[1] // ratio
    padd = np.random.randint(0, 255, (padd, im.shape[1], im.shape[2]), np.uint8)
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
    padd = np.random.randint(0, 255, (im.shape[0], padd, im.shape[2]), np.uint8)
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


def slant2(im, ratio=random.choice([30, 40])):
    ratio = 40
    selec = random.choice([-1, 1])
    roll_tally = 0
    roll_amnt = 0
    im = np.uint8(im)
    padd = im.shape[0] // ratio
    padd = np.random.randint(0, 255, (im.shape[0], padd, im.shape[2]), np.uint8)
    if selec == 1:
        im = np.concatenate((im, padd), axis=1)
    else:
        im = np.concatenate((padd, im), axis=1)
    #    padd += 255
    np.pad(im, 0, )
    for x in range(im.shape[0]):
        im[x, :] = np.roll(im[x, :], selec * roll_amnt, axis=0)
        roll_tally += 1
        roll_amnt = roll_tally // ratio
    return im


def slant_dim(img):
    img1 = slant(img, selec=1)
    img1 = slant(img1, selec=-1)
    return img1


def line_up(image):
    vert_lines = []
    horiz_lines = []
    if random.choice([True, False]):
        vert_lines.append(random.randint(0, image.shape[1] // 5))
    if random.choice([True, False]):
        vert_lines.append(random.randint((image.shape[1] * 4) // 5, image.shape[1] - 1))
    if random.choice([True, False]):
        horiz_lines.append(random.randint(0, image.shape[0] // 5))
    if random.choice([True, False]):
        horiz_lines.append(random.randint((image.shape[0] * 4) // 5, image.shape[0] - 1))
    choices = [0, 1, 2, 3, 4]
    for x_cord in vert_lines:
        for extra in range(random.choice(choices)):
            alt_coord = x_cord - extra
            if alt_coord < 0:
                pass
            else:
                if random.choice([True, False]):
                    input_arr = np.random.randint(0, 100, size=image.shape[0])
                    image[:, alt_coord] = input_arr
                else:
                    image[:, alt_coord].fill(random.uniform(0, 125))
    for y_cord in horiz_lines:
        for i, extra in enumerate(range(random.randint(2, 4))):
            alt_coord = y_cord - extra
            if alt_coord < 0:
                pass
            else:
                if random.choice([True, False]):
                    input_arr = np.random.randint(0, 100, size=image.shape[1])
                    image[alt_coord, :] = input_arr
                else:
                    image[alt_coord, :].fill(random.uniform(0, 125))

    return np.uint8(image)


if __name__ == '__main__':
    for i in range(20):
        arr = np.zeros((64, 192))
        arr.fill(255)
        im = line_up(arr)
        cv2.imshow('image', np.uint8(im))
        cv2.waitKey(0)
