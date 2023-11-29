import numpy as np
import pandas as pd
import tqdm
from torch import nn, optim
from net import FusionNet, Single_Net, LateFusionNet, Hybrid_FusionNet, Att_Fusion, get_logger, load_checkpoint, crossEntropy
import torch
import os.path
import argparse
import logging
import time
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchmetrics import MeanSquaredError
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # general
    # distributed training
    parser.add_argument('--input',
                        help='input feature type',
                        default = 'fusion',
                        type=str)
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
                        default = None,
                        type=str,
                        )
    args = parser.parse_args()
    return args

def train(net, optimizer, type, dataloader, epoches, model_path, task, logger): 
           
    # timestamp = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    timestamp = time.strftime("%Y-%m-%d-%H_%M",time.localtime(time.time()))
    logger.info('start training!')    
    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        running_mse = 0.
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):

            optimizer.zero_grad()
            
            tobii_data = x_batch[:, :11]
            ppg_data = x_batch[:, 11:]
            
            n = x_batch.shape[0]
            
            if type == 'late_fusion': 
                output = net(tobii_data.unsqueeze(2), ppg_data.unsqueeze(2)).squeeze(0)
                output = output.squeeze(1)
                # loss = crossEntropy(output, y_batch)  
            elif type == 'fusion':
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
                output = net(fusion_tp)
                output=output.squeeze(1)
                
            else:       # hybrid (tfn + fusion)
                t2 = torch.cat([tobii_data, torch.ones(n,1)], dim=1)
                p2 = torch.cat([ppg_data, torch.ones(n,1)], dim=1)
                t2 = t2.unsqueeze(2)
                p2 = p2.unsqueeze(1)
                fusion_tp = torch.einsum('nxt, nty->nxy', t2, p2)
                fusion_tp = fusion_tp.flatten(start_dim=1).unsqueeze(2)
                
                output = net(tobii_data.unsqueeze(2), ppg_data.unsqueeze(2), fusion_tp)
                output=output.squeeze(1)
                
                                     
            y_batch= y_batch.type(torch.LongTensor)  
            loss = criterion(output, y_batch)
            
            optimizer.zero_grad()    
            loss.backward()
            optimizer.step()
            
            running_loss = loss.item()
            if task == 'cla':
                accuracy = Accuracy(task="multiclass", num_classes=3)
                running_acc = accuracy(output, y_batch)     
         
        scheduler.step()
        
        if task == 'cla':
            logger.info('Epoch:[{}/{}]\t lr={:.8f}\t loss={:.5f}\t acc={:.3f}'.format(i+1, epoches, scheduler.get_last_lr()[0], running_loss, 100*running_acc))
            
        # print("[%d/%d] Loss: %.5f, Acc: %.2f" %(i+1, epoches, running_loss, 100*running_acc))
        torch.save({'epoch': i + 1, 'state_dict': net.state_dict(), 'loss': running_loss,
                'optimizer': optimizer.state_dict()},
                model_path + '/' + input + '_' + task + '_b' + str(BATCH_SIZE) + '_' + timestamp + '.pth.tar')
    
    logger.info('finish training!')
    logger.info("=> Saved model to {}".format(model_path + '/' + input + '_' + task + '_b' + str(BATCH_SIZE) + '_' + timestamp + '.pth.tar'))

        

if __name__ == "__main__":  
    
    args = parse_args()
    len_tobii = 11
    len_ppg = 9
    hidden_dim = 30
    # hidden_dim = 32
    
    input = args.input
    task = args.task
    BATCH_SIZE = args.bn
    
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

    logpath ="logs/" + now + '_' + input + '_train' + '_' + task + '_b' + str(BATCH_SIZE) + '_' + r".log" 
    logger = get_logger(logpath)
    # writer = SummaryWriter(log_dir="logs2")
    
    DataPath = 'data/train.csv'
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
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    x = x.reshape(x.shape[0], 1, x.shape[1], 1)
    y = y.reshape(y.shape[0],1)

    
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
        net = Single_Net((len_ppg+1)*(len_tobii+1), hidden_dim, output_dim) 
    elif input == 'att':
        net = Att_Fusion(len_tobii, len_ppg, hidden_dim, output_dim)
    else:
        net = Hybrid_FusionNet(len_tobii, len_ppg, 120, hidden_dim, output_dim)

    
    optimizer = optim.Adam(net.parameters(), lr = 0.00005) 
    # optimizer = optim.Adam(net.parameters(), lr = 0.00001) 
    scheduler = MultiStepLR(optimizer, milestones=[300, 400], gamma = 0.1)
    
    model_path = 'models'
    
    train(net, optimizer, input, dataloader, 500, model_path, task, logger)
    

