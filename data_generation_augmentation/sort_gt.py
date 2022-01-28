import os


dir = '/Users/petergeraghty/ocr_experiments/data_generation_augmentation/storysquad-ground-truth/'
new_dir = '/Users/petergeraghty/ocr_experiments/data_generation_augmentation/alt_gt/'

if __name__ == '__main__':
    direc = {}
    for x in os.listdir(dir):
        if x[-3:] == 'txt':
                text = open(dir+x, 'r').read()
                x = x.split('.')
                x = x[0].split('-')
                if len(x[0]) <4:
                    page_num = x[1]
                    x = x[2:]
                else:
                    page_num = x[0]
                    x = x[1:]
                for i, z in enumerate(x):
                    if len(z) > 3:
                        x.remove(z)
                if len(x) > 2:
                    continue

                # if page not in direc:
                #     direc[page] = {}
                #     direc[page][order_num] = text
                # else:
                #     direc[page][order_num] = text

    print(direc.keys())
    print(len(direc.keys()))