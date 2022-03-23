import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os

def get_bb_of_item(item):
    ''' Helper function to find the bounding box (bb) of an item in the xml file.
    All the characters within the item are found and the left-most (min) and right-most (max + length)
    are found.
    The bounding box emcompasses the left and right most characters in the x and y direction.
    Parameter
    ---------
    item: xml.etree object for a word/line/form.
    height: int
        Height of the form to calculate percentages of bounding boxes
    width: int
        Width of the form to calculate percentages of bounding boxes
    Returns
    -------
    list
        The bounding box [x, y, w, h] in percentages that encompasses the item.
    '''

    character_list = [a for a in item.iter("cmp")]
    if len(character_list) == 0:  # To account for some punctuations that have no words
        return None
    x1 = np.min([int(a.attrib['x']) for a in character_list])
    y1 = np.min([int(a.attrib['y']) for a in character_list])
    x2 = np.max([int(a.attrib['x']) + int(a.attrib['width']) for a in character_list])
    y2 = np.max([int(a.attrib['y']) + int(a.attrib['height']) for a in character_list])

    bb = [x1, y1, x2 - x1, y2 - y1]
    return bb

def crop_image(image, bb):
    ''' Helper function to crop the image by the bounding box (in percentages)
    '''
    (x1, y1, x2, y2) = bb
    x2 = x1 + x2
    y2 = y1 + y2
    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
    return image[y1:y2, x1:x2]

if __name__ == '__main__':
    im_dir = '/Users/petergeraghty/ocr_experiments/iam_data/forms/'
    xml_dir = '/Users/petergeraghty/ocr_experiments/iam_data/xml/'
    dir = '/Users/petergeraghty/ocr_experiments/subset_data_2/'
    # num = 0
    # for doc in os.listdir(xml_dir):
    #     tree = ET.parse(xml_dir + doc)
    #     root = tree.getroot()
    #     image = cv2.imread(im_dir+doc[:-3] + 'png', 0)
    #     for x in root.iter('handwritten-part'):
    #         for i, y in enumerate(x.iter('word')):
    #             text = y.attrib['text']
    #             alpha = get_bb_of_item(y)
    #             if alpha is None:
    #                 continue
    #             width, height = alpha[2:]
    #             alpha = crop_image(image, alpha)
    #             cv2.imwrite(dir+str(num)+'_'+text+'_'+str(width)+'_'+str(height)+'.png', alpha)


# items saved in format: item-number_text_width_height.png