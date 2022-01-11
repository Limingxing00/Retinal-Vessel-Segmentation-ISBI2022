import cv2
import os
import pdb
import random
import yaml
import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from lib.config import parse_args
from loader.load_data import MyDataset
from loss_function import DiceLoss
# from loss_vgg import PerceptualLoss

from network.twonet import Dual_net

from eval import metrics
from evalutate.inference import infer
# from evalutate.metrics import metrics


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# if not os.path.isdir("./results/events"):
    # os.mkdir("./results/events")

# writer = SummaryWriter("./results/events")
global_score = 0.7


class Crop_Resize(object):
    def __init__(self, scale):
        """
        :param scale:  a list [0.5, 1.0]
        """

        # self.size = tuple(reversed(size))  # size: (n, c, h, w)
        self.scale_pkg = scale

    def __call__(self, img, label):
        self.scale = random.sample(self.scale_pkg, 1)[0]

        # pdb.set_trace()
        _, _, h, w = img.shape
        h_security = range(int(h - self.scale * h))  # 282
        w_security = range(int(w - self.scale * w))  # 292
        seed_w = random.sample(w_security, 1)[0]
        seed_h = random.sample(h_security, 1)[0]
        img_security = (img)[:, :, seed_h: seed_h + int(self.scale * h),
                       seed_w: int(seed_w + self.scale * w)]
        label_security = (label)[:, :, seed_h: seed_h + int(self.scale * h),
                         seed_w: int(seed_w + self.scale * w)]
        # pdb.set_trace()

        # pdb.set_trace()
        # check porint
        # cv2.imwrite("img.tiff", np.array(img))
        # cv2.imwrite("label.tiff", np.array(label))
        return img_security, label_security


def inverse_freq(label):
    den = label.sum()  # 0
    _, _, h, w = label.shape
    num = h * w
    alpha = den / num  # 0
    return torch.tensor([alpha, 1 - alpha]).cuda()



def chk_break(model, checkpoint_PATH, LOAD_MODEL=None):
    if LOAD_MODEL:
        model_CKPT = torch.load(checkpoint_PATH)  # original model path
        model.load_state_dict(model_CKPT)
        print('loading checkpoint!', checkpoint_PATH)

        return model
    else:
        return True


# def multiloss(out, x1, x2, x3, label):
#     # pdb.set_trace()
#
#     label05 = F.upsample(label, size=(int(label.size(2) // 32 * 32 * 0.5), int(label.size(3) // 32 * 32 * 0.5)))
#     label025 = F.upsample(label, size=(int(label.size(2) // 32 * 32 * 0.25), int(label.size(3) // 32 * 32 * 0.25)))
#     label0125 = F.upsample(label, size=(int(label.size(2) // 32 * 32 * 0.125), int(label.size(3) // 32 * 32 * 0.125)))
#     loss = loss_func(out, label.long())
#     loss05 = loss_func(x1, label05.long())
#     loss025 = loss_func(x2, label025.long())
#     loss0125 = loss_func(x3, label0125.long())
#     loss = loss + loss05 + loss025 + loss0125
#
#     celoss = loss_func2(out, label.squeeze(1).long())
#     celoss05 = loss_func2(x1, label05.squeeze(1).long())
#     celoss025 = loss_func2(x2, label025.squeeze(1).long())
#     celoss0125 = loss_func2(x3, label0125.squeeze(1).long())
#     loss2 = celoss + celoss05 + celoss025 + celoss0125
#     return loss, loss2

def cal_foreground_ratio(label):
    """
    calculate the ratio of the foreground.
    """
    N, _, H, W = label.shape
    ratio = []
    assert label.max()==1
    for n in range(N):
        mol = label[n,...].sum()
        den = H*W
        ratio.append(mol/den)
    ratio = np.array(ratio)
    # pdb.set_trace()
    assert ratio.max()<=1, "Please check label ratio!"
    return np.array(ratio)
    

def train(epoch):
    model.train()
    batch_id = 0
    for data, label in train_dataset:
        data, label = data.to(device), label.to(device)
        
        # # pre-process
        # data = preprocess(data)
        data, label = Crop_Resize(data, label)
        ratio = cal_foreground_ratio(label)

        optimizer.zero_grad()
        x1, x2 = model(data, ratio)

        # pdb.set_trace()

        dice = dice_loss(x1, label.long())
        l1 = smooth_l1(x2, x1)
        # l1_2 = smooth_l1(x3, x1)

        loss = dice+cfg["RCE_WEIGHT"]*l1

        loss.backward()

        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        batch_id += 1
        if batch_id % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, dice {:.4f}, l1 {:.4f}'.format(
                epoch, batch_id * len(data), cfg['BATCH_SIZE'] * len(train_dataset),
                       100. * batch_id / len(train_dataset), loss.item(), dice.item(), l1.item()))

        # writer.add_scalar('train/dice', dice.item(), epoch * len(train_dataset) + batch_id * cfg['BATCH_SIZE'])
        # writer.add_scalar('train/l1', l1.item(), epoch * len(train_dataset) + batch_id * cfg['BATCH_SIZE'])


def model_checkpoint(model, epoch, f1, acc, auc, specificity, precision, sensitivity, cfg):
    global global_score
    # if score > global_score:
    print("Epoch:{}. New logging is loading...".format(epoch))
    # pdb.set_trace()

    fw = open(cfg["LOG_PATH"], 'a')

    fw.write(
        "model number:{}, f1 {:.4f}, auc {:.4f}, acc {:.4f}, specificity {:.4f}, precision {:.4f}, sensitivity {:.4f}.".format(
            epoch, f1, auc, acc, specificity, precision, sensitivity))

    fw.write("\n")
    fw.close()
    
    # save the model if f1 is greater than global score
    if f1>global_score:
        try:
            torch.save(model.state_dict(), cfg['MODEL_PATH'] + "/" + "epoch_{}_f1_{:.4f}.pth".format(epoch, f1))
        except:
            torch.save(model.module.state_dict(), cfg['MODEL_PATH'] + "/" + "epoch_{}_f1_{:.4f}.pth".format(epoch, f1))
        global_score = f1
    # print("Now, the score is {}".format(global_score))


def view_prediction(path, i, is_val=True):
    view_name = os.listdir(path)
    assert len(view_name) == 1
    img = np.array(Image.open(path + "/" + view_name[0]))
    # pdb.set_trace()
    # if is_val:
        # writer.add_image('val', img, i, walltime=0, dataformats="HW")
    # else:
        # writer.add_image('train', img, i, walltime=0, dataformats="HW")


def init_dir():
    # if not os.path.exists(cfg['VIEW_TRAIN_PATH']):
    # os.makedirs(cfg['VIEW_TRAIN_PATH'])
    # if not os.path.exists(cfg['VIEW_VAL_PATH']):
    # os.makedirs(cfg['VIEW_VAL_PATH'])
    if not os.path.exists(cfg['TEST_PRED_PATH']):
        os.makedirs(cfg['TEST_PRED_PATH'])
    if not os.path.exists(cfg['MODEL_PATH']):
        os.makedirs(cfg['MODEL_PATH'])


if __name__ == "__main__":
    args = parse_args()
    f = open(args.cfg_file)
    cfg = yaml.load(f)

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dual_net(cfg).to(device)
    Crop_Resize = Crop_Resize(cfg["multi_scale"])


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.to(device)

    chk_break(model, cfg['BREAKPOINT'], cfg["IS_BREAKPOINT"])
    optimizer = optim.Adam(model.parameters(), lr=cfg['LEARNING_RATE'])
    

    dice_loss = DiceLoss().to(device)
    smooth_l1 = nn.SmoothL1Loss().to(device)
    # perceptualLoss = PerceptualLoss()

    train_dataloader = MyDataset(data_path=cfg['TRAIN_DATA_PATH'],
                                 label_path=cfg['TRAIN_LABEL_PATH'],
                                 transform=transforms.Compose([transforms.ToTensor()]))

    train_dataset = DataLoader(train_dataloader, batch_size=cfg['BATCH_SIZE'], shuffle=True)

    init_dir()
    # lambda2 = lambda epoch: 0.997 ** epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)

    # initialize your own logging
    fw = open(cfg["LOG_PATH"], 'w')
    fw.close()

    for i in range(cfg['EPOCH']):
        model.train()
        train(i)
        # scheduler.step()

        if i % cfg['CHECK_BATCH'] == 0:
            # model_checkpoint(model, i)

            model.eval()
            infer(model=model,
                  device=device,
                  cfg=cfg)

            f1, acc, auc, specificity, precision, sensitivity = metrics(label_path=cfg['TEST_LABEL_PATH'],
                                                                        prediction_path=cfg['TEST_PRED_PATH'],
                                                                        cfg=cfg)
            # writer.add_scalar('metric/f1', f1, i)
            # writer.add_scalar('metric/auc', auc, i)

            model_checkpoint(model, i, f1, acc, auc, specificity, precision, sensitivity, cfg)
