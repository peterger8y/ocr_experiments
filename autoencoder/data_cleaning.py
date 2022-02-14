import os
import cv2

dir = '/Users/petergeraghty/ocr_experiments/crop_cloud_data_gen/'
new_dir = '/Users/petergeraghty/ocr_experiments/subset_data/'

if __name__ == '__main__':
    for i, file in enumerate(os.listdir(dir)):
        if i % 5 == 0:
            image = cv2.imread(dir+file, 0)
            cv2.imwrite(new_dir + file, image)