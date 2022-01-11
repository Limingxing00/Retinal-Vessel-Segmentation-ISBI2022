# -*- coding: utf-8 -*-
# @Time    : 2019/11/19 17:00
# @Author  : Mingxing Li
# @FileName: metrics.py
# @Software: PyCharm
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, roc_auc_score
import os
import yaml
import pdb
from lib.config import parse_args


def metrics(label_path, prediction_path, foreground=None):
    """
    :param foreground: pixel value 255 is foreground.
    """
    # pdb.set_trace()
    label_file_name = sorted(os.listdir(label_path))
    pred_file_name = sorted(os.listdir(prediction_path))
    f1m = []
    aucm = []
    # pdb.set_trace()
    for i in range(len(label_file_name)):

        label = Image.open(label_path + "/" + label_file_name[i])
        label = np.array(label)
        pred = Image.open(prediction_path + "/" + pred_file_name[i])
        pred = (np.array(pred)).flatten() / 255
        label = (label).astype(np.uint8).flatten() / 255
        # pdb.set_trace()

        # check the pixel value
        try:
            assert label.max() == 1 and (pred).max() <= 1
            assert label.min() == 0 and (pred).min() >= 0

            if foreground:
                # pdb.set_trace()
                fpr, tpr, thresholds = roc_curve((label), pred)
                AUC_ROC = roc_auc_score(label, pred)

                pred[pred < 0.5] = 0
                pred[pred >= 0.5] = 1
                f1 = f1_score(label, pred)
            else:
                f1 = f1_score(1 - label, 1 - pred)
        except:
            print("The prediction is not good.")
            return 0

        # print(classification_report(label, pred, target_names=["class 0", "class 1"]))
        f1m.append(f1)
        aucm.append(AUC_ROC)
    print("Your score of new data is {}".format(np.array(f1m).mean()))
    return np.array(f1m).mean(), np.array(aucm).mean()


if __name__ == "__main__":
    args = parse_args()
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    m = metrics(label_path=cfg['TEST_DATA_PATH'], prediction_path=cfg['TEST_PRED_PATH'])
    print(m)
