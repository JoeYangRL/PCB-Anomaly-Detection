import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from build_dataset import Mydataset, Mydataset_test, split_data, cal_mean_and_std, read_and_preprocess, read_and_preprocess_test
import matplotlib.pyplot as plt
import datetime
from build_model import Net1, get_parameter_number
from main_contrast import contrast
try:
    import cPickle
except:
    import _pickle as cPickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_path = r"E:\research\PCB Anomaly Detection\data\rebuild_datas_0906"

def accuracy(output1, output2, target, threshold=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    oo = F.softmax(output1,dim=1)
    #print(oo[0])
    o1 = output1.tolist()
    t = target.tolist()
    if threshold == None:
        for i in range(len(t)):
            if t[i] == 1:  # anomaly
                if o1[i].index(max(o1[i])) == t[i]:
                    TP += 1
                else:
                    FN += 1
            else:  # normal
                if o1[i].index(max(o1[i])) == t[i]:
                    TN += 1
                else:
                    FP += 1
    else:
        for i in range(len(t)):
            if t[i] == 1:  # anomaly
                if oo[i][1] > threshold:
                    TP += 1
                else:
                    FN += 1
            else:  # normal
                if oo[i][1] <= threshold:
                    TN += 1
                else:
                    FP += 1

    return TP, FP, TN, FN

def test(model,data_loader,epoch):

    model.eval()
    model.mode = 'eval'
    acc_all = []
    pre_all = []
    recall_all = []
    fpr_all = []
    a = [i / 100.0 for i in range(0,100,1)]
    print(a)

    for j in a:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i, data in enumerate(data_loader, 0):
            x_q, x_ref, y, y_ref = data
            x_q = x_q.cuda()
            x_ref = x_ref.cuda()
            y = y.cuda()
            logits, yq, yk = model(x_q, x_ref)
            # print('y_all:',y_all)
            # print('y_truth:',y_truth)
            tp, fp, tn, fn = accuracy(yq, yk, y, j)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
        acc_all.append((TP + TN) / (TP + TN + FP + FN))
        pre_all.append(TP / (TP + FP))
        if TP + FN != 0:
            recall_all.append(TP / (TP + FN))  # most important, tpr
        else:
            recall_all.append(0)
        if FP + TN != 0:
            fpr_all.append(FP / (FP + TN))
        else:
            fpr_all.append(0)
        print(j)

    return acc_all, pre_all, recall_all, fpr_all

def cal_auc(fpr,tpr):

    auc = 0
    for i in range(len(fpr)-1):
        if fpr[i+1] - fpr[i] != 0:
            auc -= 0.5 * (tpr[i+1]+tpr[i]) * (fpr[i+1] - fpr[i])

    return auc

def main(path, pth_name):


    batch_size = 16
    mean, std = cal_mean_and_std(path + '/train.txt', image_path)
    augmentation2 = [
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    test_transform = transforms.Compose(augmentation2)
    img_trans_t, img_ref_trans2_t, l_t, l_ref_t = read_and_preprocess_test(path + '/test.txt', image_path,
                                                                           transform=test_transform)

    test_data = Mydataset_test(img_trans_t, img_ref_trans2_t, l_t, l_ref_t)

    test_loader = DataLoader(test_data, batch_size=int(batch_size / 2), shuffle=True, drop_last=False)
    m = torch.load(path + pth_name)
    acc, pre, recall, fpr = test(m, test_loader, 1)
    print(acc, '\n', pre, '\n', recall, '\n', fpr)
    """pre.insert(0,0)
    pre.insert(-1,1)
    recall.insert(0,1)
    recall.insert(-1,0)"""
    fpr.insert(-1, 0)
    recall.insert(-1, 0)
    auc = cal_auc(fpr, recall)
    plt.plot(fpr, recall, label="roc")
    plt.title('AUC = %.6f' % auc)
    # plt.plot(np.linspace(0, 1, 9998), np.linspace(0, 1, 9998))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.show()


if __name__ == '__main__':


    path = 'logs/20201120205918'
    pth_name = '/contrast_model_epoch_82_of_300.pth'
    main(path, pth_name)

    """AUC += 0.5 * (TPR + TPR_last) * (FPR - FPR_last)"""