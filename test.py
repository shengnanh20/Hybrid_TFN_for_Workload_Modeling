from genericpath import exists
from pickle import TRUE
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from sklearn.metrics import f1_score, roc_auc_score
from torchmetrics.classification import Accuracy, MulticlassF1Score, Recall, AUROC

import tqdm
from torch import nn, optim
from net import FusionNet, Single_Net, LateFusionNet, Hybrid_FusionNet, Att_Fusion, get_logger, load_checkpoint
import torch
import os.path
import argparse
import time
from torchmetrics import MeanSquaredError

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import RocCurveDisplay

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    # general
    # distributed training
    parser.add_argument('--input',
                        help='input feature type',
                        default = 'fusion',
                        type=str,
                        )
    parser.add_argument('--task',
                        help='training task',
                        default = 'cla',
                        type=str,
                        )
    parser.add_argument('--bn',
                        help='batch size',
                        default = '32',
                        type=int,
                        )
    parser.add_argument('--path',
                        help='model path',
                        type=str,
                        )
    args = parser.parse_args()
    return args


def test(net, type, x_test, y_test, task, logger):
    running_acc = 0.
    running_mse = 0.
    for (j, input_data) in enumerate(x_test):
        
        if task == 'cla':
            label = (y_test[j]).long()
        else:
            label = y_test[j]
        
            
        tobii_data = input_data[:, :11, :]
        ppg_data = input_data[:, 11:, :]
        
        if type == 'fusion':
            output = net(tobii_data, ppg_data).squeeze(0)
        elif type == 'late_fusion': 
            output = net(tobii_data, ppg_data).squeeze(0)
        elif type == 'pre_fusion':
            output = net(input_data).squeeze(0)
        elif type == 'ppg':
            output = net(ppg_data).squeeze(0)
        elif type == 'tobii':
            output = net(tobii_data).squeeze(0)
        else:
            t2 = torch.cat([tobii_data, torch.ones(1,1,1)], dim=1)
            p2 = torch.cat([ppg_data, torch.ones(1,1,1)], dim=1).permute([0,2,1])
            fusion_tp = torch.einsum('nxt, nty->nxy', t2, p2)
            fusion_tp = fusion_tp.flatten(start_dim=1).unsqueeze(2)
            output = net(fusion_tp).squeeze(0)
                
        if task == 'cla':
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.data.item()
        
        else:
            output = output.squeeze(0)
            mse = MeanSquaredError()
            running_mse += mse(output, label).data.item()
     
    if task == 'cla':     
        running_acc /= len(x_test)
        logger.info('Testing acc={:.3f}'.format(running_acc))
        
        
    else:
        running_mse /= len(x_test)
        logger.info('Testing mse={:.3f}'.format(running_mse))
        
    # logger.info("F1-Score:{:.4f}".format(f1_score(label_all,prob_all)))
    
            
    # print("Test Acc: %.2f" %( 100*running_acc))

def test2(net, type, dataloder, task, logger):
    running_acc = 0.
    running_mse = 0.
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            
        tobii_data = x_batch[:, :11]
        ppg_data = x_batch[:, 11:]
        n = x_batch.shape[0]
        
        if type == 'fusion':
            output = net(tobii_data.unsqueeze(2), ppg_data.unsqueeze(2)).squeeze(0)
            output = output.squeeze(1)
        elif type == 'late_fusion': 
            output = net(tobii_data.unsqueeze(2), ppg_data.unsqueeze(2)).squeeze(0)
            output = output.squeeze(1)
        elif type == 'pre_fusion':
            output = net(x_batch.unsqueeze(2)).squeeze(0)
            output = output.squeeze(1)
        elif type == 'ppg':
            output = net(ppg_data.unsqueeze(2)).squeeze(0)
            output = output.squeeze(1)
        elif type == 'tobii':
            output = net(tobii_data.unsqueeze(2)).squeeze(0)
            output = output.squeeze(1)
        elif input == 'att':
            output = net(tobii_data.unsqueeze(2), ppg_data.unsqueeze(2)).squeeze(0)
            output=output.squeeze(1)
        elif type == 'tfn_ppg':
                t2 = torch.cat([ppg_data, torch.ones(n,1)], dim=1)
                p2 = torch.cat([ppg_data, torch.ones(n,1)], dim=1)
                t2 = t2.unsqueeze(2)
                p2 = p2.unsqueeze(1)
                fusion_tp = torch.einsum('nxt, nty->nxy', t2, p2)
                fusion_tp = fusion_tp.flatten(start_dim=1).unsqueeze(2)
                output = net(fusion_tp)
                output=output.squeeze(1)
        elif type == 'tfn_tobii':
                t2 = torch.cat([tobii_data, torch.ones(n,1)], dim=1)
                p2 = torch.cat([tobii_data, torch.ones(n,1)], dim=1)
                t2 = t2.unsqueeze(2)
                p2 = p2.unsqueeze(1)
                fusion_tp = torch.einsum('nxt, nty->nxy', t2, p2)
                fusion_tp = fusion_tp.flatten(start_dim=1).unsqueeze(2)
                output = net(fusion_tp)
                output=output.squeeze(1)      
        elif type == 'tfn':
            t2 = torch.cat([tobii_data, torch.ones(n,1)], dim=1)
            p2 = torch.cat([ppg_data, torch.ones(n,1)], dim=1)
            t2 = t2.unsqueeze(2)
            p2 = p2.unsqueeze(1)
            fusion_tp = torch.einsum('nxt, nty->nxy', t2, p2)
            fusion_tp = fusion_tp.flatten(start_dim=1).unsqueeze(2)
            output = net(fusion_tp).squeeze(0)
            output=output.squeeze(1)            
        else:
            t2 = torch.cat([tobii_data, torch.ones(n,1)], dim=1)
            p2 = torch.cat([ppg_data, torch.ones(n,1)], dim=1)
            t2 = t2.unsqueeze(2)
            p2 = p2.unsqueeze(1)
            fusion_tp = torch.einsum('nxt, nty->nxy', t2, p2)
            fusion_tp = fusion_tp.flatten(start_dim=1).unsqueeze(2)
                
            output = net(tobii_data.unsqueeze(2), ppg_data.unsqueeze(2), fusion_tp).squeeze(0)
            output=output.squeeze(1)
            
                
        y_batch= y_batch.type(torch.LongTensor)  
        
        accuracy = Accuracy(task="multiclass", num_classes=3)
        running_acc = accuracy(output, y_batch) 
        logger.info('Testing acc={:.3f}'.format(running_acc))    
        
        f1s = MulticlassF1Score(num_classes=3)
        f1s = f1s(output, y_batch)
        logger.info('F1 score={:.3f}'.format(f1s))
        
        recall = Recall(task="multiclass", average='macro', num_classes=3)
        recall = recall(output, y_batch)
        logger.info('Recall={:.3f}'.format(recall))
        
        auroc = AUROC(task="multiclass", num_classes=3)
        auroc = auroc(output, y_batch)
        logger.info('AUROC={:.3f}'.format(auroc))
        
        
    # logger.info("F1-Score:{:.4f}".format(f1_score(label_all,prob_all)))
    
            
    # print("Test Acc: %.2f" %( 100*running_acc))
          

if __name__ == "__main__":  
    
    args = parse_args()
    
    len_tobii = 11
    len_ppg = 9
    hidden_dim = 30
    # hidden_dim = 32
    output_dim = 3
    input = args.input
    task = args.task
    model_path = args.path
    
    
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
    logpath ="logs/" + now + '_' + input + '_valid' + '_' + task + r".log" 
    logger = get_logger(logpath)
    
    DataPath = 'data/test.csv'
    data = pd.read_csv(DataPath)
    data = data.fillna(0)
    # data_group(data)

    x = data.iloc[:, 3:23]
    if task == 'reg':
        y = data.iloc[:, 24]
        criterion = torch.nn.MSELoss(reduction='mean')
        output_dim = 1
    else:
        y = data.iloc[:, 23] - 1
        criterion = torch.nn.CrossEntropyLoss()
        output_dim = 3

    x = np.array(x)
    x = torch.tensor(x, dtype=torch.float32)
    y = np.array(y)
    y = torch.tensor(y, dtype=torch.float32)
    
    BATCH_SIZE = x.shape[0]
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    x = x.reshape(x.shape[0], 1, x.shape[1], 1)
    y = y.reshape(y.shape[0],1)

    # input = 'ppg'

    if input == 'fusion':
        net = FusionNet(len_tobii, len_ppg, hidden_dim, output_dim)    
    elif input == 'pre_fusion':
        net = Single_Net(len_tobii + len_ppg, hidden_dim, output_dim)   
    elif input == 'late_fusion':
        net = LateFusionNet(len_tobii, len_ppg, hidden_dim, output_dim)     
    elif input == 'ppg':
        net = Single_Net(len_ppg, hidden_dim, output_dim)  
    elif input == 'tobii':
        net = Single_Net(len_tobii, hidden_dim, output_dim)  
    elif input == 'tfn_ppg':
        net = Single_Net((len_ppg+1)**2, hidden_dim, output_dim)  
    elif input == 'tfn_tobii':
        net = Single_Net((len_tobii+1)**2, hidden_dim, output_dim)   
    elif input == 'tfn':
        net = Single_Net(120, hidden_dim, output_dim)   
    elif input == 'att':
        net = Att_Fusion(len_tobii, len_ppg, hidden_dim, output_dim)        
    else:
        net = Hybrid_FusionNet(len_tobii, len_ppg, 120, hidden_dim, output_dim)

    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0001)

    # model_path = '/home/shengnan/hpe/hp/models'
    # logger.info("=> loading model from {}".format(model_path))
    
    net, optimizer = load_checkpoint(net, model_path, optimizer)
    logger.info("=> loading model from {}".format(model_path))
    # net = torch.load(model_path)
    # test(net, input, x, y, task, logger)
    test2(net, input, dataloader, task, logger)

