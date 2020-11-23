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
try:
    import cPickle
except:
    import _pickle as cPickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_path = r"E:\research\PCB Anomaly Detection\data\rebuild_datas_0906"

#hyper parameters
batch_size = 64
epoch = 300
lr0 = 0.001
schedule = [70, 120] #lr *= 0.1
momentum = 0.9 #for SGD
weight_decay = 1e-4
es_threshold = 50

def accuracy_val(output, output_ref, target, target_ref):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    count = 0
    count_ref = 0
    o = output.tolist()
    o_ref = output_ref.tolist()
    t = target.tolist()
    t_ref = target_ref.tolist()
    for i in range(len(t)):
        if o[i].index(max(o[i])) == t[i]:
            count += 1
        if o_ref[i].index(max(o_ref[i])) == t_ref[i]:
            count_ref += 1

    return count / len(target), count_ref / len(target)

def accuracy_1(output1, output2, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    count = 0
    o1 = output1.tolist()
    o2 = output2.tolist()
    t = target.tolist()
    for i in range(len(t)):
        if o1[i].index(max(o1[i])) == t[i]:
            count += 1
        if o2[i].index(max(o2[i])) == t[i]:
            count += 1
    return count / (2 * len(target))

class contrast(nn.Module):


    def __init__(self,net, T=0.07):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        stage_out_channels: output channel number of each stage in ShuffuleNet v2
        stage_repeats: how many times a block repeated within a stage of ShuffleNet v2
        """
        super(contrast, self).__init__()

        self.T = T
        self.mode = 'train'

        # create the encoders
        self.encoder = net

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        fq,yq = self.encoder(im_q)  # queries: NxC
        q = nn.functional.normalize(fq, dim=1)
        fk,yk = self.encoder(im_k)  # queries: NxC
        k = nn.functional.normalize(fk, dim=1)

        #print(q_all.size(),k_all.size())
        # compute logits
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q,k])
        #print(logits.size())

        # apply temperature
        logits /= self.T

        #print(y_all.size())

        return logits, yq, yk

def train(train_loader,model,lr0,epoch,schedule,val_loader, path, es_threshold):

    warning_flag = 0
    model.cuda()
    model.train()
    model.mode = 'train'
    loss1 = nn.CrossEntropyLoss().cuda()
    loss2 = nn.CrossEntropyLoss().cuda()
    lr = lr0
    run_record = open(path + '/run_record.txt', mode='a')
    a1, a2 = val(model,val_loader,0,run_record)
    best_val_acc = a1 + a2
    early_stop_count = 0
    for j in range(epoch):
        if j+1 in schedule:
            lr = lr * 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)
        loss_sum = 0
        loss1_sum = 0
        loss2_sum = 0
        acc_sum = 0
        for i, data in enumerate(train_loader, 0):
            x_q, x_k, y= data
            x_q = x_q.cuda()
            x_k = x_k.cuda()
            y = y.cuda()
            logits, yq, yk = model(x_q, x_k)
            labels = torch.Tensor([k for k in range(batch_size)]).cuda()
            labels = labels.long()

            acc = accuracy_1(yq, yk, y)
            #print(acc)
            """for k in range(batch_size):
                if y[k*2] == 1:
                    loss += loss1()"""
            #loss = loss1(logits, labels) + 2 * loss2(yq, y) + 2* loss2(yk, y)
            contrast_loss = loss1(logits, labels)
            class_loss = 0
            for k in range(int(len(y)/2)):
                class_loss += 2 * (loss2(yq[2*k:(2*k+1)], y[2*k:2*k+1]) + loss2(yk[2*k:2*k+1], y[2*k:2*k+1])) / 2
                class_loss += 1 * (loss2(yq[2*k+1:2*(k+1)], y[2*k+1:2*(k+1)]) + loss2(yk[2*k+1:2*(k+1)], y[2*k+1:2*(k+1)])) / 2
            #print('loss2(yk,y) ',loss2(yk,y)+loss2(yq,y))
            class_loss  = class_loss / (len(y)/2)
            #print('class_loss ', class_loss)
            loss = 2 * class_loss + 1 * contrast_loss
            #loss = loss1(logits, labels)
            #loss = loss2(y_all, y_truth)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
            loss1_sum += contrast_loss
            loss2_sum += class_loss
            acc_sum += acc
            #print('finish batch %i epoch %i' % (i, j))
        loss_mean = loss_sum / (i + 1)
        loss1_mean = loss1_sum / (i + 1)
        loss2_mean = loss2_sum / (i + 1)
        acc_mean = acc_sum / (i + 1)
        print('Train Epoch: {}\t Total Loss: {:.6f} Contrast Loss: {:.6f} Classification Loss: {:.6f} Acc: {:.6f}'.format(
            (j+1), loss_mean.item(), loss1_mean.item(), loss2_mean.item(), acc_mean))
        run_record.write('Train Epoch: {}\t Total Loss: {:.6f} Contrast Loss: {:.6f} Classification Loss: {:.6f} Acc: {:.6f}'.format(
            (j+1), loss_mean.item(), loss1_mean.item(), loss2_mean.item(), acc_mean) + '\n')

        val_acc, val_ref_acc = val(model,val_loader,j+1,run_record)
        if val_acc + val_ref_acc <= best_val_acc:
            early_stop_count += 1
        else:
            early_stop_count = 0
            save_model = model
            save_epoch = j+1
            best_val_acc = val_acc + val_ref_acc

        if early_stop_count >= es_threshold and (j+1) > schedule[-1] + es_threshold and warning_flag == 1:
            print('early stop!')
            break
        elif early_stop_count >= es_threshold and (j+1) > schedule[-1] + es_threshold:
            lr = lr * 0.1
            early_stop_count = 0
            warning_flag = 1
    torch.save(save_model, path + '/contrast_model_epoch_%s_of_%s.pth' % (save_epoch,epoch))
    run_record.write('Training finished with best model at Epoch: {}\t val_Acc: {:.6f}'.format(save_epoch, best_val_acc/2) + '\n')


def val(model,data_loader,epoch,run_record):

    model.eval()
    model.mode = 'eval'
    acc_sum = 0
    acc_sum_ref = 0
    for i, data in enumerate(data_loader, 0):
        x_q, x_k, y, y_ref= data
        x_q = x_q.cuda()
        x_k = x_k.cuda()
        y = y.cuda()
        y_ref = y_ref.cuda()
        logits, yq, yk = model(x_q, x_k)
        #print('y_all:',y_all)
        #print('y_truth:',y_truth)
        acc, acc_ref = accuracy_val(yq, yk, y, y_ref)
        acc_sum += acc
        acc_sum_ref += acc_ref
    acc_mean = acc_sum / (i + 1)
    acc_ref_mean = acc_sum_ref / (i+1)
    print('After Training Epoch: {}\t val_Acc: {:.6f}  val_Acc_ref: {:.6f}'.format(epoch, acc_mean, acc_ref_mean))
    run_record.write('After Training Epoch: {}\t val_Acc: {:.6f} val_Acc_ref: {:.6f}'.format(epoch, acc_mean, acc_ref_mean) + '\n')
    model.train()
    model.mode = 'train'

    return acc_mean, acc_ref_mean

def main():

    t = datetime.datetime.now()
    timestamp = '%0*i%0*i%0*i%0*i%0*i%0*i' % (4, t.year, 2, t.month, 2, t.day, 2, t.hour, 2, t.minute, 2, t.second)

    path = 'logs/%s'%timestamp
    os.mkdir(path)
    split_data(r'E:\research\PCB Anomaly Detection\data\rebuild_datas_0906\label.txt', timestamp, save_txt_path=path)
    mean, std = cal_mean_and_std(path + '/train.txt', image_path)
    augmentation1 = [
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    train_transform = transforms.Compose(augmentation1)
    img_trans1, img_trans2, l = read_and_preprocess(path + '/train.txt', image_path,
                                                    transform=train_transform)
    train_data = Mydataset(img_trans1, img_trans2, l)
    """with open(path + '/train_data.pickle', 'wb') as fp:
        cPickle.dump(train_data, fp)"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    """for i, data in enumerate(train_loader):
        x_q, x_k, y = data
        img = x_q[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
        img = img.numpy()  # FloatTensor转为ndarray
        img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
        plt.imshow(img)
        plt.show()"""

    augmentation2 = [
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    test_transform = transforms.Compose(augmentation2)
    img_trans_t, img_ref_trans_t, l_t, l_ref_t = read_and_preprocess_test(path + '/test.txt', image_path,
                                                          transform=test_transform)

    test_data = Mydataset_test(img_trans_t, img_ref_trans_t, l_t, l_ref_t)
    """with open(path + '/test_data.pickle', 'wb') as fp:
        cPickle.dump(test_data, fp)"""
    test_loader = DataLoader(test_data, batch_size=int(batch_size/2), shuffle=True, drop_last=True)
    print('data prepared.')

    net = Net1([24, 48, 96, 192, 1024], [1, 1, 1])
    f1 = open(path+ '/net_structure.txt', 'a')
    f1.write(str(net)+ '\n')
    total_num, trainable_num = get_parameter_number(net)
    f1.write('Total:' + str(total_num) + ' Trainable:' + str(trainable_num))
    m = contrast(net)
    print('model created.')
    with open(path + '/training_paras_record.txt', "a") as fp:
        fp.write('batch_size = %s' % batch_size + ' ' + 'epoch = %s' % epoch + ' ' + 'lr0 = %s' % lr0 + ' ' + 'schedule = %s' % schedule + ' '
                 'momentum = %s' % momentum + ' ' + 'weight_decay = %s' % weight_decay + ' ' + 'es_threshold = %s' % es_threshold + '\n')
    train(train_loader, m, lr0, epoch, schedule, test_loader, path, es_threshold)

if __name__ == '__main__':
    # load data

    main()