from pyts.approximation import DiscreteFourierTransform
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from scipy import signal
import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2


# Tactile feature pre-processing
def dimension_reduction(sensor_type: str):
    output_dir = './outputdata/tactile/'
    os.makedirs(output_dir, exist_ok=True)
 
    for n in range(1, 64):
        obj_prefix = f'object_0{n}' if n < 10 else f'object_{n}'
        
        for m in range(1, 6):
            for batch in ['NF', 'NP', 'R8LO', 'R8RO', 'R8RU']:
                feel_path = f'./dataset/{obj_prefix}/observation_{m}/tactile_{sensor_type}_noise/Tactile_{sensor_type}_{batch}.csv'
                back_path = f'./dataset/{obj_prefix}/Background/{sensor_type}/Background_{sensor_type}_{batch}.csv'
                out_path = f'{output_dir}/{obj_prefix}_{m}_{sensor_type}{batch}.csv'

                readData1 = pd.read_csv(feel_path, header=None)
                readData2 = pd.read_csv(back_path, header=None)
                subresult = readData1 - readData2
                result = signal.resample(subresult, 10000, axis=0)

                pd.DataFrame(result).to_csv(out_path, index=False, header=None)

def concatenate_tactile_data():
    in_path = './outputdata/tactile/*.csv'
    csv_list = glob.glob(in_path)
    csv_train = []
    csv_val = []
    csv_test = []
    # add tag
    for file in csv_list:
        table = str(int(file.split("_")[2]) - 1)
        f = open(file,'r+',encoding='utf-8')
        content = f.read()
        f.seek(0, 0)
        f.write(table + '\n' + content)
    #     f.close()

    # concatenate
    for file in csv_list:
        df = pd.read_csv(file, header=None)
        data = df.values
        data = list(map(list, zip(*data)))
        data = pd.DataFrame(data)
        if len(csv_train) == 0:
            csv_train = data
        else:
            if int(file.split("_")[3]) == 3:
                print(file)
                if len(csv_test) == 0:
                    csv_test = data
                else:
                    csv_test = pd.concat([csv_test, data], axis=0)
            else:
                csv_train = pd.concat([csv_train, data], axis=0)
    
    csv_test.reset_index(drop=True, inplace=True)
    csv_val = csv_test.iloc[::2]
    csv_test = csv_test.iloc[1::2]

    csv_train = csv_train.sort_values(by=0, ascending=True).reset_index(drop=True)
    csv_val = csv_val.sort_values(by=0, ascending=True).reset_index(drop=True)
    csv_test = csv_test.sort_values(by=0, ascending=True).reset_index(drop=True)

    csv_train.to_csv('./outputdata/t_train.csv', index=False, header=None)
    csv_val.to_csv('./outputdata/t_val.csv', index=False, header=None)
    csv_test.to_csv('./outputdata/t_test.csv', index=False, header=None)


def kinesthetic_extract():
    output_dir = './outputdata/kinesthetic/'
    os.makedirs(output_dir, exist_ok=True)

    for n in range(1, 64):
        obj_prefix = f'object_0{n}' if n < 10 else f'object_{n}'

        for m in range(1, 6):
            kine_path = './dataset/' + str(obj_prefix) + '/observation_' + str(m) + '/kinesthetics/'
            files = [os.path.join(kine_path, file) for file in os.listdir(kine_path)]  # 一个oj的一个ob【有序】
            df = pd.DataFrame(index=[0])
            cnt = 0
            for file in files:
                print(file)
                cnt += 1
                readData1 = pd.read_csv(file, header=None)
                df=df.append(readData1)
                if cnt % 12 ==0:
                    out_path = output_dir + str(obj_prefix) +'_'+ str(m) +'_kinesthetic.csv'
                    df = df.dropna(axis=0)
                    df.to_csv(out_path, index=False, header=None)


def concatenate_kinesthetic_data():
    in_path = './outputdata/kinesthetic/*.csv'
    csv_list = glob.glob(in_path)
    csv_train = []
    csv_test = []
    # add tag
    for file in csv_list:
        table = str(int(file.split("_")[1]) - 1)
        f = open(file,'r+',encoding='utf-8')
        content = f.read()
        f.seek(0, 0)
        f.write(table + '\n' + content)
        f.close()

    # concatenate
    for file in csv_list:
        df = pd.read_csv(file, header=None)
        data = df.values
        data = list(map(list, zip(*data)))
        data = pd.DataFrame(data)
        if len(csv_train) == 0:
            csv_train = data
        else:
            if int(file.split("_")[2]) == 3:
                if len(csv_test) == 0:
                    csv_test = data
                else:
                    csv_test = pd.concat([csv_test, data], axis=0)
            else:
                csv_train = pd.concat([csv_train, data], axis=0)
    csv_test = pd.DataFrame(np.tile(csv_test, (10, 1)))
    csv_train = pd.DataFrame(np.tile(csv_train, (10, 1)))

    csv_test.reset_index(drop=True, inplace=True)
    csv_val = csv_test.iloc[::2]
    csv_test = csv_test.iloc[1::2]

    csv_train = csv_train.sort_values(by=0, ascending=True).reset_index(drop=True)
    csv_val = csv_val.sort_values(by=0, ascending=True).reset_index(drop=True)
    csv_test = csv_test.sort_values(by=0, ascending=True).reset_index(drop=True)

    csv_train.to_csv('./outputdata/k_train.csv', index=False, header=None)
    csv_val.to_csv('./outputdata/k_val.csv', index=False, header=None)
    csv_test.to_csv('./outputdata/k_test.csv', index=False, header=None)


def calculate_mean_std(train_path):
    img_h, img_w = 224, 224
    means, stdevs = [], []
    img_list = []
    
    imgs_patt_list = []
    folder_path = os.path.join(train_path)
    imgs = os.listdir(folder_path)
    for img in imgs:
        imgs_patt_list.append(os.path.join(folder_path, img))
    len_ = len(imgs_patt_list)
    
    i = 0
    for item in imgs_patt_list:
        img = cv2.imread(item, -1)
        img = cv2.resize(img, (img_w, img_h))
        img = np.reshape(img, (224, 224, -1))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):  # rgb
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR --> RGB
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


def RegNet(mean=[0.47956696, 0.48325223, 0.4745792], std=[0.18494284, 0.18796188, 0.18584234], train_path='v_train', val_path='v_val', test_path='v_test'):
    transform1 = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    featuremodel = models.regnet_y_32gf(weights='IMAGENET1K_SWAG_E2E_V1', progress=True)
    featuremodel.fc = nn.Linear(3712, 1000)
    torch.nn.init.eye(featuremodel.fc.weight)

    for param in featuremodel.parameters():
        param.requires_grad = False

    if val_path != None:
        data_val = []
        tag_val = []
        for filename in os.listdir(val_path):
            tag = int(filename.split("_")[1]) - 1
            print(filename, '\n')
            img = Image.open(val_path + '/' + filename)
            img1 = transform1(img)

            x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
            featuremodel.training=False
            y = featuremodel(x)
            y = y.data.numpy()
            tag_val.append(tag)
            data_val.append(y)
        np.save(val_path + 'v_val.csv', data_val)
        np.save(val_path + 'vtag_val.csv', tag_val)
    
    if test_path != None:
        data_test = []
        tag_test = []
        for filename in os.listdir(test_path):
            tag = int(filename.split("_")[1]) - 1
            print(filename, '\n')
            img = Image.open(test_path + '/' + filename)
            img1 = transform1(img)

            x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
            featuremodel.training=False
            y = featuremodel(x)
            y = y.data.numpy()
            tag_test.append(tag)
            data_test.append(y)
        np.save(test_path + 'v_test.csv', data_test)
        np.save(test_path + 'vtag_test.csv', tag_test)
    
    if train_path != None:
        data_train = []
        tag_train = []
        for filename in os.listdir(train_path):
            tag = int(filename.split("_")[1]) - 1
            print(filename, '\n')
            img = Image.open(train_path + '/' + filename)
            img1 = transform1(img)

            x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
            featuremodel.training=False
            y = featuremodel(x)
            y = y.data.numpy()
            tag_train.append(tag)
            data_train.append(y)
        np.save(train_path + 'v_train.csv', data_train)
        np.save(train_path + 'vtag_train.csv', tag_train)


def tagdata2csv(tagpath, datapath, topath, rows, columns):
    tagfile = np.load(tagpath)
    tagfile = np.array(tagfile)
    tagfile = np.reshape(tagfile, (rows, 1))
    tag_to_csv = pd.DataFrame(data=tagfile)

    datafile = np.load(datapath)
    datafile = np.array(datafile)
    datafile = np.reshape(datafile, (rows, columns))
    np_to_csv = pd.DataFrame(data=datafile)

    merge_csv = pd.concat([tag_to_csv, np_to_csv], axis=1)
    merge_csv = merge_csv.sort_values(by=0, ascending=True).reset_index(drop=True)
    merge_csv.to_csv(topath, index=False, header=False)

if __name__ == '__main__':
    # # Tactile feature pre-processing
    dimension_reduction('feel')
    dimension_reduction('poke')
    concatenate_tactile_data()

    # Kinesthetic feature pre-processing
    kinesthetic_extract()
    concatenate_kinesthetic_data()

    # # Visual feature pre-processing
    base_path = '/outputdata/visual/'
    train_mean, train_std = calculate_mean_std(base_path+'train') 
    RegNet(mean=train_mean.tolist(), std=train_std.tolist(), train_path=base_path+'train', val_path=base_path+'val', test_path=base_path+'test')
    tagdata2csv(tagpath=base_path+"valvtag_val.csv.npy", datapath=base_path+"valv_val.csv.npy", topath=base_path+"v_val.csv", rows=315, columns=1000)
    tagdata2csv(tagpath=base_path+"testvtag_test.csv.npy", datapath=base_path+"testv_test.csv.npy", topath=base_path+"v_test.csv", rows=315, columns=1000)
    tagdata2csv(tagpath=base_path+"trainvtag_train.csv.npy", datapath=base_path+"trainv_train.csv.npy", topath=base_path+"v_train.csv", rows=2520, columns=1000)