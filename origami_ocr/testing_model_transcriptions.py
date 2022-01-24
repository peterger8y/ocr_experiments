import time
import torch
import torch.utils.data
import gin
from origami_ocr import ds_load

from utils import CTCLabelConverter, ModelEma
from origami_ocr.cnv_model import OrigamiNet
from validation import validation

device = torch.device('cpu')
pO = None


@gin.configurable
def evaluate(test_data_path, test_data_list, val_batch_size, workers):
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
    valid_dataset = ds_load.myLoadDS(test_data_list, test_data_path,
                                     ralph=ralph_vals)



    #model.cuda(opt.rank)
    model_ema = ModelEma(model)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    converter = CTCLabelConverter(ralph_vals, default=True)

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        valid_loss, current_accuracy, current_norm_ED, ted, bleu, preds, labels, infer_time = validation(
            model_ema.ema, criterion, valid_dataset, converter, None, pO)

    v_time = time.time() - start_time

    out = f' Loss: {valid_loss.item():0.5f} time: ({v_time:0.1f})'
    out += f' vloss: {valid_loss:0.3f}'
    out += f' CER: {ted:0.4f} NER: {current_norm_ED:0.4f}'
    out += f' B: {bleu * 100:0.2f}'
    print(out)


if __name__ == '__main__':
    gin.parse_config_file('gin_configurations/test.gin')
    evaluate()