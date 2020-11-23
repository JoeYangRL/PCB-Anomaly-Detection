from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import time
import imageio
import datetime
try:
    import cPickle
except:
    import _pickle as cPickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

filepath = r"E:\research\PCB Anomaly Detection\data\rebuild_datas_0906"
label_file = r'E:\research\PCB Anomaly Detection\data\rebuild_datas_0906\label.txt'
train_file = r'E:\research\PCB Anomaly Detection\code\logs\test-20201115171133.txt'

def split_data(label_file,timestamp,save_txt_path='',val_prop=0.1,test_prop=0.2):

    """
    train_prop = 1 - test_prop
    val_data sample from test_data
    """
    train = open(save_txt_path + '/train.txt',mode='a')
    val = open(save_txt_path + '/val.txt',mode='a')
    test = open(save_txt_path + '/test.txt',mode='a')
    #file_list = np.random.shuffle([i for i in os.listdir(filepath) if i.endswith('c.png')])
    f = open(label_file, 'r')
    for line in f:
        #print('anomaly')
        sep = line.strip('\n').split(' ')
        if sep[0].endswith('c.png') and int(sep[1]) == 1:
            r = np.random.randint(0, 1000)
            if r < val_prop * 1000:
                test.write(line)
                val.write(line)
            elif r < test_prop * 1000:
                test.write(line)
            else:
                train.write(line)
    f.seek(0)
    for line in f:
        #print('normal')
        sep = line.strip('\n').split(' ')
        if sep[0].endswith('c.png') and int(sep[1]) == 0:
            r = np.random.randint(0, 1000)
            if r < val_prop * 1000:
                test.write(line)
                val.write(line)
            elif r < test_prop * 1000:
                test.write(line)
            else:
                train.write(line)

def cal_mean_and_std(file,image_path):

    f = open(file, 'r')
    img = []
    mean = [0,0,0]
    std = [0,0,0]
    for line in f:
        line = line.strip('\n')
        sep = line.split()
        # print(sep[0])
        # print(sep[1])
        img.append(Image.open(os.path.join(image_path, sep[0])))
        img.append(Image.open(os.path.join(image_path, sep[0].split('.')[0] + '_t.png')))

    for im in img:
        for i in range(3):
            mean[i] += np.asarray(im)[:, :, i].mean()
            std[i] += np.asarray(im)[:, :, i].std()
    mean = np.asarray(mean) / len(img) / 255
    std = np.asarray(std) / len(img) / 255

    return mean, std

def read_and_preprocess(file,image_path,transform=transforms.ToTensor()):

    f = open(file, 'r')
    img = []
    l = []
    for line in f:
        line = line.strip('\n')
        sep = line.split()
        # print(sep[0])
        # print(sep[1])
        img.append(Image.open(os.path.join(image_path, sep[0])))
        img.append(Image.open(os.path.join(image_path, sep[0].split('.')[0] + '_t.png')))
        l.append(int(sep[1]))
        l.append(0)
    img_trans1 = []
    img_trans2 = []
    for im in img:
        img_trans1.append(transform(im))
        img_trans2.append(transform(im))

    return img_trans1, img_trans2, l

def read_and_preprocess_test(file,image_path,transform=transforms.ToTensor()):

    f = open(file, 'r')
    img = []
    img_ref = []
    l = []
    l_ref = []
    for line in f:
        line = line.strip('\n')
        sep = line.split()
        # print(sep[0])
        # print(sep[1])
        img.append(Image.open(os.path.join(image_path, sep[0])))
        img_ref.append(Image.open(os.path.join(image_path, sep[0].split('.')[0] + '_t.png')))
        l.append(int(sep[1]))
        l_ref.append(0)
    img_trans = []
    img_ref_trans = []
    for im,im_ref in zip(img, img_ref):
        img_trans.append(transform(im))
        img_ref_trans.append(transform(im_ref))

    return img_trans, img_ref_trans, l, l_ref

class Mydataset(Dataset):
    def __init__(self,img_trans1, img_trans2, l):

        super(Mydataset,self).__init__()

        imginfo = []
        for i in range(len(img_trans1)):
            imginfo.append((img_trans1[i], img_trans2[i], l[i]))
        self.imginfo = imginfo

    def __getitem__(self, idx):

        img_q, img_k, label = self.imginfo[idx]

        return img_q, img_k, label

    def __len__(self):

        return len(self.imginfo)

class Mydataset_test(Dataset):
    def __init__(self,img_trans, img_ref_trans, l, l_ref):

        super(Mydataset_test,self).__init__()

        imginfo = []
        for i in range(len(img_trans)):
            imginfo.append((img_trans[i], img_ref_trans[i], l[i], l_ref[i]))
        self.imginfo = imginfo

    def __getitem__(self, idx):

        im, im_ref, label, label_ref = self.imginfo[idx]

        return im, im_ref, label, label_ref

    def __len__(self):

        return len(self.imginfo)

if __name__ == '__main__':

    #split_data(label_file,save_txt_path='logs')
    """augmentation = [
        #transforms.ToTensor(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        #transforms.Normalize(mean=mean, std=var)
    ]
    transform = transforms.Compose(augmentation)
    test = Mydataset(train_file,filepath,transform)
    print(test.mean, test.var)
    print(test.__len__())
    test_loader = DataLoader(test, batch_size=1, shuffle=True, drop_last=True)
    for i, data in enumerate(test_loader, 0):
        x, x_ref, y, y_ref = data
        print(y)
        print(y_ref)"""
    m,s = cal_mean_and_std(train_file,filepath)
    print(m,s)