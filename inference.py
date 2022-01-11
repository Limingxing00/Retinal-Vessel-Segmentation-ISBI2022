import numpy as np
import cv2
import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from network.twonet import Dual_net
import torch.nn.functional as F
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
from network.unetpp import NestedUNet
import yaml
from lib.config import parse_args
import pdb
import time

transform = transforms.Compose([
    transforms.ToTensor(),
])


def do_overlap(data, model, stride=0, roi_h=512, roi_w=512):
    # pdb.set_trace()
    _, _, w, h = data.shape
    # pdb.set_trace()
    # assert w==512 and h==512
    # assert roi_h == roi_w
    # assert (h - roi_h) % stride == 0
    output = torch.zeros(1, 1, w, h)
    frequency = torch.zeros(1, 1, w, h)

    number = 1  # int((h - roi_h) / stride + 1)

    # predict
    # pdb.set_trace()
    pred = model(data, ratio=[1])[0]
    infer_time = model(data, ratio=[1])[-1]
    pred = F.softmax(pred, dim=1)
    pred = pred[0, 1, ...].cpu()

    # pred = weight_mul(pred)


    # output[output > 0.5] = 1
    # output[output <= 0.5] = 0

    return pred, infer_time


def infer(model, device, cfg):
    data_path=cfg['TEST_DATA_PATH']
    prediction_path=cfg['TEST_PRED_PATH']

    file_name = sorted(os.listdir(data_path))
    # pdb.set_trace()
    with torch.no_grad():
        time = []
        for i in range(len(file_name)):
            data = Image.open(data_path + "/" + file_name[i])
            
            if cfg["INPUT_CHANNEL"]==1:
                data = torch.from_numpy(np.array(data)).unsqueeze(0).float() / 255
            elif cfg["INPUT_CHANNEL"]==3:
                data = torch.from_numpy(np.array(data).transpose(2, 0, 1)).float() / 255
            else:
                raise RuntimeError('Please check input channel of the dataset.')
            # data = preprocess(data)

            data = data.to(device).unsqueeze(0)

            pred, infer_time = do_overlap(data, model, 0)
            time.append(infer_time)
            pred = pred.cpu().numpy()
            pred = pred * 255
            # pdb.set_trace()
            pred = Image.fromarray(np.uint8(pred))

            pred.save(prediction_path + "/" + str(i + 1) + "_refine_1st_" + file_name[i])
            # pdb.set_trace()

            # pred = do_overlap(data, model, 1)
            # pred = pred.cpu().numpy()
            # pred = pred * 255
            # pred = 255 - pred
            # # pdb.set_trace()
            # pred = Image.fromarray(np.uint8(pred))

            # pred.save(prediction_path + "/" + str(i + 1) + "_refine_2nd_" + file_name[i])

            # pred = do_overlap(data, model, 2)
            # pred = pred.cpu().numpy()
            # pred = pred * 255
            # pred = 255 - pred
            # # pdb.set_trace()
            # pred = Image.fromarray(np.uint8(pred))

            # pred.save(prediction_path + "/" + str(i + 1) + "_1st_" + file_name[i])

            # pred = do_overlap(data, model, 3)
            # pred = pred.cpu().numpy()
            # pred = pred * 255
            # pred = 255 - pred
            # # pdb.set_trace()
            # pred = Image.fromarray(np.uint8(pred))

            # pred.save(prediction_path + "/" + str(i + 1) + "_2nd_" + file_name[i])

            print("{}-th over".format(i))

    time = np.array(time)
    print("time", time.mean())

if __name__ == "__main__":
    args = parse_args()
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    model_num = cfg['MODEL_NUMBER']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dual_net(cfg).cuda()
    # pdb.set_trace()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.to(device)

    model.load_state_dict(torch.load(cfg['MODEL_PATH'] + "/" + "{}.pth".format(model_num)))
    model.eval()

    if not os.path.isdir(cfg['TEST_PRED_PATH']):
        os.mkdir(cfg['TEST_PRED_PATH'])

    infer(model=model,
          device=device,
          cfg=cfg)
