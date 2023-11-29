import torch
from torch import nn
import torch.nn.functional as F


class Att_Fusion(nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_dim, output):
        super(Att_Fusion, self).__init__()
        self.conv1 = nn.Conv1d(in_channels1, hidden_dim, 1)
        self.conv2 = nn.Conv1d(in_channels2, hidden_dim, 1)
        self.pool = nn.MaxPool1d(3, 2)
        self.softmax = nn.Softmax(dim=2)
        # self.fc = nn.Linear(28, output)
        self.fc = nn.Linear(hidden_dim, output)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim , nhead=6, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=1)
        
    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1)).permute([0,2,1])
        # x1 = self.pool(x1.permute([0,2,1]))
        x2 = F.relu(self.conv2(x2)).permute([0,2,1])
        # x2 = self.pool(x2.permute(0,2,1))
        
        out_features = self.transformer_decoder(x1, x2)            
        out_features =  out_features  / ( out_features.norm(dim=-1, keepdim=True) + 1e-5)     

        out = self.fc( out_features)
        out = self.softmax(out)
        return out

class FusionNet(nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_dim, output):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels1, hidden_dim, 1)
        self.conv2 = nn.Conv1d(in_channels2, hidden_dim, 1)
        self.pool = nn.MaxPool1d(3, 2)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(28, output)
        
    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool(x1.permute([0,2,1]))

        x2 = F.relu(self.conv2(x2))
        x2 = self.pool(x2.permute(0,2,1))
        
        feat = torch.cat([x1, x2], dim=2)
        out = self.fc(feat)
        # out = self.softmax(out)
        return out

class LateFusionNet(nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_dim, output):
        super(LateFusionNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels1, hidden_dim, 1)
        self.conv2 = nn.Conv1d(in_channels2, hidden_dim, 1)
        self.pool1 = nn.MaxPool1d(3, 2)
        self.pool2 = nn.MaxPool1d(3, 2)
        self.softmax = nn.Softmax(dim=2)
        self.fc1 = nn.Linear(14, output)
        self.fc2 = nn.Linear(14, output)
        self.weight = 0.8
        
    def forward(self, x1, x2):
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool1(x1.permute([0,2,1]))
        x1 = self.fc1(x1)
        x1 = self.softmax(x1)

        x2 = F.relu(self.conv2(x2))
        x2 = self.pool2(x2.permute([0,2,1]))
        x2 = self.fc2(x2)
        x2 = self.softmax(x2)
        
        prob = x1 * self.weight + x2 * (1 - self.weight)
        # prob = (x1 + x2) / 2
        
        return prob


# class Single_Net(nn.Module):
#     def __init__(self, in_channels, hidden_dim, output):
#         super(Single_Net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, hidden_dim, 1)
#         self.conv2 = nn.Conv1d(hidden_dim, 64, 1)
#         self.dropout1 = nn.Dropout1d(0.25)
#         self.pool = nn.MaxPool1d(3, 2)
#         self.softmax = nn.Softmax(dim=2)
#         # self.fc = nn.Linear(14, output)
#         self.fc = nn.Linear(31, output)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x.permute([0,2,1]))
#         x = self.dropout1(x)
#         out = self.fc(x)
#         # out = self.softmax(out)
#         return out

class Single_Net(nn.Module):
    def __init__(self, in_channels, hidden_dim, output):
        super(Single_Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, 1)
        self.pool = nn.MaxPool1d(3, 2)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(14, output)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x.permute([0,2,1]))
        out = self.fc(x)
        # out = self.softmax(out)
        return out

class Hybrid_FusionNet(nn.Module):
    def __init__(self, in_channels1, in_channels2, tf_channels, hidden_dim, output):
        super(Hybrid_FusionNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels1, hidden_dim, 1)
        self.conv2 = nn.Conv1d(in_channels2, hidden_dim, 1)
        self.pool = nn.MaxPool1d(3, 2)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Linear(42, output)
        
        self.conv3 = nn.Conv1d(tf_channels, hidden_dim, 1)
        

    def forward(self, x1, x2, x_tf):
        x1 = F.relu(self.conv1(x1))
        x1 = self.pool(x1.permute([0,2,1]))

        x2 = F.relu(self.conv2(x2))
        x2 = self.pool(x2.permute(0,2,1))
        
        x_tf = F.relu(self.conv3(x_tf))
        x_tf = self.pool(x_tf.permute([0,2,1]))
        
        feat = torch.cat([x1, x2, x_tf], dim=2)
        out = self.fc(feat)
        out = self.softmax(out)
        return out


def data_group(data):
    ind_list = []
    id = 1
    for index in range(len(data)):
        if index == 0:
            ind_list.append(id)
            continue
        if data.iloc[index]['Person'] == data.iloc[index - 1]['Person'] and data.iloc[index]['Trial'] == data.iloc[index - 1]['Trial']:
            ind_list.append(id)
        else:
            id += 1
            ind_list.append(id) 
    data.insert(data.shape[1], 'Index', ind_list)
    data.to_csv("t2.csv", index=True)
    return 


import logging
 
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


def crossEntropy(logits, y_true):
    c = -torch.log(logits.gather(1, y_true.reshape(-1, 1)) + 1e-4)
    return torch.sum(c)
           