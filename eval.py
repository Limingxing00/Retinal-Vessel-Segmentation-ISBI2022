import numpy as np
from matplotlib import pyplot as plt

# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score


import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, roc_auc_score
import os
import yaml
import pdb
from lib.config import parse_args
import warnings

warnings.filterwarnings("ignore")

"""
calculate metrics for entire retinal vessel images.
"""


def metrics(label_path, prediction_path, cfg):
    """
    :param foreground: pixel value 255 is foreground.
    """
    label_file_name = sorted(os.listdir(label_path))
    pred_file_name = sorted(os.listdir(prediction_path))
    f1m = []
    accm = []
    aucm = []
    specificitym = []
    precisionm = []
    sensitivitym = []

    # pdb.set_trace()
    for i in range(len(label_file_name)):
        label = Image.open(label_path + "/" + label_file_name[i])
        label = np.array(label)

        # label[label <= 128] = 0
        # label[label > 128] = 1

        pred = Image.open(prediction_path + "/" + pred_file_name[i])
        pred = (np.array(pred)).flatten() / 255
        if label.max()==1:
            label = (label).astype(np.uint8).flatten()
        elif label.max()==255:
            label = (label).astype(np.uint8).flatten() / 255
        else:
            raise RuntimeError('Please check your label.')
        # pdb.set_trace()

        # check the pixel value
        # pdb.set_trace()

        assert label.max() == 1 and (pred).max() <= 1
        assert label.min() == 0 and (pred).min() >= 0


        # test another datasets ISBI 2012
        if cfg['DATASET'] == "ISBI2012":
            label = 1 - label
            pred = 1 - pred


        y_scores, y_true = pred, label

        # Area under the ROC curve
        # pdb.set_trace()
        fpr, tpr, thresholds = roc_curve((y_true), y_scores)
        AUC_ROC = roc_auc_score(y_true, y_scores)
        # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
        # print ("\nArea under the ROC curve: " +str(AUC_ROC))

        # ap_score = average_precision_score(y_true, y_scores)
        # Precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        AUC_prec_rec = np.trapz(precision, recall)
        # print ("\nArea under Precision-Recall curve: " +str(AUC_prec_rec))

        # Confusion matrix
        threshold_confusion = 0.5
        # print ("\nConfusion matrix:  Custom threshold (for positive) of " +str(threshold_confusion))
        y_pred = np.empty((y_scores.shape[0]))
        for i in range(y_scores.shape[0]):
            if y_scores[i] >= threshold_confusion:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        confusion = confusion_matrix(y_true, y_pred)
        # print (confusion)
        accuracy = 0
        if float(np.sum(confusion)) != 0:
            accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        # print ("Global Accuracy: " +str(accuracy))
        specificity = 0
        if float(confusion[0, 0] + confusion[0, 1]) != 0:
            specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        # print ("Specificity: " +str(specificity))
        sensitivity = 0
        if float(confusion[1, 1] + confusion[1, 0]) != 0:
            sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        # print ("Sensitivity: " +str(sensitivity))
        precision = 0
        if float(confusion[1, 1] + confusion[0, 1]) != 0:
            precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
        # print ("Precision: " +str(precision))

        # Jaccard similarity index
        # jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
        # print ("\nJaccard similarity score: " +str(jaccard_index))

        # F1 score
        F1_score = f1_score(y_true, y_pred, average='binary')
        # print ("\nF1 score (F-measure): " +str(F1_score))
        # print(1)



        # print(classification_report(label, pred, target_names=["class 0", "class 1"]))
        f1m.append(F1_score)
        accm.append(accuracy)
        aucm.append(AUC_ROC)
        specificitym.append(specificity)
        precisionm.append(precision)
        sensitivitym.append(sensitivity)

    # print("Your score of new data is {}".format(np.array(f1m).mean()))
    return np.array(f1m).mean(), np.array(accm).mean(), np.array(aucm).mean(), np.array(specificitym).mean(), np.array(
        precisionm).mean(), np.array(sensitivitym).mean()


if __name__ == "__main__":
    args = parse_args()
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    # pdb.set_trace()
    f1, acc, auc, specificity, precision, sensitivity = metrics(label_path=cfg['TEST_LABEL_PATH'],
                                                                prediction_path=cfg['TEST_PRED_PATH'],
                                                                cfg=cfg)
    print("f1", f1, "accuracy", acc, "auc", auc,
          "specificity", specificity, "precision", precision,
          "sensitivity", sensitivity)

# ====== Evaluate the results
