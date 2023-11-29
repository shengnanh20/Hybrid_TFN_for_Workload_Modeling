from genericpath import exists
from pickle import TRUE
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import tqdm
from torch import nn, optim
from net import FusionNet, Single_Net
import torch
import os.path
import argparse
import logging

def train(net, type, epoches, model_path):
    if os.path.isfile(model_path):
        net = torch.load(model_path)
            
    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (j, input_data) in enumerate(x_train):
            label = (y_train[j] - 1).long()
            optimizer.zero_grad()
            
            tobii_data = input_data[:, :11, :]
            ppg_data = input_data[:, 11:, :]
            
            if type == 'fusion':
                output = net(tobii_data, ppg_data).squeeze(0)
            elif type == 'ppg':
                output = net(ppg_data).squeeze(0)
            else:
                output = net(tobii_data).squeeze(0)
                
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.data.item()
            
        running_loss /= len(x_train)
        running_acc /= len(x_train)
        torch.save(net, model_path)     
        print("[%d/%d] Loss: %.5f, Acc: %.2f" %(i+1, epoches, running_loss, 100*running_acc))
    

def test(net, type, x_test, y_test):
    running_acc = 0.
    for (j, input_data) in enumerate(x_test):
        label = (y_test[j] - 1).long()
            
        tobii_data = input_data[:, :11, :]
        ppg_data = input_data[:, 11:, :]
        if type == 'fusion':
            output = net(tobii_data, ppg_data).squeeze(0)
        elif type == 'ppg':
            output = net(ppg_data).squeeze(0)
        else:
            output = net(tobii_data).squeeze(0)

        _, predict = torch.max(output, 1)
        correct_num = (predict == label).sum()
        running_acc += correct_num.data.item()
        
    running_acc /= len(x_test)
            
    print("Test Acc: %.2f" %( 100*running_acc))
          

if __name__ == "__main__":  
    
    DataPath = 'data/dataSet.csv'
    DataFolder = 'data/'
    data = pd.read_csv(DataPath)
    data = data.fillna(0)
    # data_group(data)

    x = data.iloc[:, 3:23]
    y = data.iloc[:, 23]

    # splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state = 7)

    # split = splitter.split(data, groups=data['Index'])
    # train_ix, test_ix = next(split)

    # x_train, x_test, y_train, y_test =  x.loc[train_ix], x.loc[test_ix], y.loc[train_ix], y.loc[test_ix]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    x_train = np.array(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = np.array(y_train)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = np.array(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = np.array(y_test)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], 1)

    y_train = y_train.reshape(y_train.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)

    
    len_tobii = 11
    len_ppg = 9
    hidden_dim = 30
    output_dim = 3
    # input = 'tobii'
    input = 'ppg'

    if input == 'fusion':
        net = FusionNet(len_tobii, len_ppg, hidden_dim, output_dim)       
    elif input == 'ppg':
        net = Single_Net(len_ppg, hidden_dim, output_dim)             
    else:
        net = Single_Net(len_tobii, hidden_dim, output_dim)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    model_path = 'models/model_' + input + '.pth.tar'
    
    # train(net, input, 300, model_path)
    net = torch.load(model_path)
    test(net, input, x_test, y_test)

