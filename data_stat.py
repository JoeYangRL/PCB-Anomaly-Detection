import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import math
try:
    import cPickle
except:
    import _pickle as cPickle

filepath = r"E:\research\PCB Anomaly Detection\data\rebuild_datas_0906"

def count_mask(filepath):
    count = np.zeros([300, 300])
    for filename in os.listdir(filepath):
        if filename.endswith('label.png'):
            a = cv2.imread(filepath + '/' + filename)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if a[i][j][0] == 255:
                        count[i][j] += 1
            print(filename)

    with open('count_mask.pickle', 'wb') as fp:
        cPickle.dump(count, fp)


"""with open('count_mask.pickle', 'rb') as fp:
    count = cPickle.load(fp)"""

def count_any_mask(count):
    count_any = np.zeros([300, 300])
    for i in range(count.shape[0]):
        for j in range(count.shape[1]):
            if count[i][j] > 0:
                count_any[i][j] = 255
    with open('count_any_mask.pickle', 'wb') as fp:
        cPickle.dump(count_any, fp)

def make_label(filepath, txt_name):

    """
    1 for anomaly, 0 for normal
    """
    normal = 0
    anomaly = 0
    pair = 0
    normal_pair = 0
    for filename in os.listdir(filepath):
        if filename.endswith('c.png'):
            pair += 1
            a = cv2.imread(filepath + '/' + filename.split('_')[0] + '_label.png')
            if a.sum() == 0:
                with open(txt_name,"a") as fp:
                    fp.write(filename + " 0\n") #normal
                    normal += 1
                    fp.write(filename.split('_')[0] + "_c_t.png" + " 0\n")
                    normal += 1
                    normal_pair += 1
            else:
                with open(txt_name,"a") as fp:
                    fp.write(filename + " 1\n") #anomaly
                    anomaly += 1
                    fp.write(filename.split('_')[0] + "_c_t.png" + " 0\n")
                    normal += 1
    anomaly_pair = pair - normal_pair
    with open(txt_name, "a") as fp:
        fp.write("total_normal_data " + str(normal) + '\n')
        fp.write("total_anomaly_data " + str(anomaly) + '\n')
        fp.write("total_pair " + str(pair) + '\n')
        fp.write("total_normal_pair " + str(normal_pair) + '\n')
        fp.write("total_anomaly_pair " + str(anomaly_pair) + '\n')

if __name__ == '__main__':
    make_label(filepath,r"E:\research\PCB Anomaly Detection\data\rebuild_datas_0906\label.txt")