import time
import torch
import torch.utils.data
import gin
import numpy as np
import skimage
import os
from PIL import Image
import boto3
import botocore
from origami_ocr import ds_load
from dotenv import load_dotenv
import argparse

from utils import CTCLabelConverter, ModelEma
from origami_ocr.cnv_model import OrigamiNet
from validation import validation

device = torch.device('cpu')
pO = None
load_dotenv()


def get_images(fname, max_w=750, max_h=750, nch=1):

    image_data = np.array(Image.open(fname))
    image_data = np.array(Image.fromarray(image_data).resize((max_h, max_w)))
    image_data = skimage.img_as_float32(image_data)

    h, w = np.shape(image_data)[:2]
    if image_data.ndim < 3:
        image_data = np.expand_dims(image_data, axis=-1)

    if nch == 3 and image_data.shape[2] != 3:
        image_data = np.tile(image_data, 3)

    image_data = np.pad(image_data, ((0, 0), (0, max_w - np.shape(image_data)[1]), (0, 0)), mode='constant',
                        constant_values=(1.0))
    return image_data



@gin.configurable
def evaluate(test_data_path, test_data_list, val_batch_size, workers, impath='None'):
    """Function to evaluate transcriptions of Origami model"""

    checkpoint = torch.load('saved_models/best_norm_ED-2.pth',
                            map_location=torch.device('cpu'))
    state_dict = {}
    for key, item in checkpoint['model'].items():
        state_dict[key[7:]] = item
    model = OrigamiNet()
    model.load_state_dict(state_dict)
    model.train()
    print('------------------------\n',
          'model loaded and ready!\n',
          '-=-=-=-=-=-=-=-=-=-=-=\n')
    ralph_vals = [' ', '!', '"', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6',
                  '7', '8', '9', ':', ';', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                  'H', 'I', 'J', 'K', 'L', 'M', 'N',
                  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                  'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                  'z']


    #model.cuda(opt.rank)
    model_ema = ModelEma(model)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    converter = CTCLabelConverter(ralph_vals, default=True)
    start_time = time.time()
    model.eval()

    if impath=='None':
        valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path,
                                         ralph=ralph_vals)
        with torch.no_grad():
            valid_loss, current_accuracy, current_norm_ED, ted, bleu, preds, labels, infer_time = validation(
                model_ema.ema, criterion, valid_dataset, converter, None, pO)

        v_time = time.time() - start_time

        out = f' Loss: {valid_loss.item():0.5f} time: ({v_time:0.1f})'
        out += f' vloss: {valid_loss:0.3f}'
        out += f' CER: {ted:0.4f} NER: {current_norm_ED:0.4f}'
        out += f' B: {bleu * 100:0.2f}'
        print(out)
    else:
        image = get_images(impath)
        print('image loaded')
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=1)
        dat = torch.tensor(image)
        image = dat.to(device)
        alpha = model(image, '')
        print('producing prediction....')
        preds_size = torch.IntTensor([alpha.size(1)] * 1)
        preds = alpha.permute(1, 0, 2).log_softmax(2)
        _, preds_index = preds.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
        preds_str = converter.decode(preds_index.data, preds_size.data)
        v_time = time.time() - start_time
        print('time for transcription: ', v_time, ' seconds')
        print(preds_str[0])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--impath", default="None",
                        help="provide image path for transcirption, if none provided\nfolder in gin file will be used"
                             "for evaluation")
    args = parser.parse_args()
    impath = args.impath
    if not os.path.exists('saved_models/best_norm_ED-2.pth'):
        print('loading model...')
        BUCKET_NAME = os.getenv('BUCKET_NAME')
        PATH = os.getenv('PATH_LOC')
        DIR_LOC = os.getenv('DIR_LOC')
        s3 = boto3.resource('s3') #config=Config(signature_version=UNSIGNED))

        try:
            s3.Bucket(BUCKET_NAME).download_file(PATH, DIR_LOC)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
    else:
        print('model already in place...')
    gin.parse_config_file('gin_configurations/test.gin')
    evaluate(impath=impath)


