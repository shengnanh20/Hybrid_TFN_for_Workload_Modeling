import numpy as np
import pandas as pd

from sklearn                 import metrics
from sklearn.linear_model    import LogisticRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torchmetrics.classification import Accuracy, MulticlassF1Score, Recall, AUROC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

PARAMETERS = {
    'elasticnet': {
        "C": [1e-3, 1e-1, 1, 10, 1000], 
        "class_weight": ['balanced', None],
        "max_iter": [100, 500, 1000],
        "l1_ratio": [0.01, 0.1, 0.3, 0.5, 0.9, 0.99]
    }
}

def load_data(path, task):
    data = pd.read_csv(path)
    data = data.fillna(0)

    x = data.iloc[:, 3:23]
    if task == 'reg':
        y = data.iloc[:, 24]
    else:
        y = data.iloc[:, 23] - 1
    
    x = np.array(x)
    x = torch.tensor(x, dtype=torch.float32)  
    tobii_data = x[:, :11]
    ppg_data = x[:, 11:]
    
    y = np.array(y)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y        


# def grid_search(x_dev, y_dev):
#     clf = LogisticRegression(
#         penalty='elasticnet', solver='saga', random_state=0
#     )
#     param_grid = PARAMETERS['elasticnet']
#     gsc = GridSearchCV(
#         estimator=clf,
#         param_grid=param_grid,
#         scoring='roc_auc'
#     )
#     gsc.fit(x_dev, y_dev)

#     return gsc.best_estimator_


def main():
    # get features and labels
    train_data = 'data/train.csv'
    test_data = 'data/test.csv'
    task = 'cla'
    
    x_train, y_train = load_data(train_data, task)
    x_test, y_test = load_data(test_data, task)

    # predict
    # model = ElasticNet(alpha=0.01, normalize=True)
    model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # model = LogisticRegression(random_state=0)
    # model = GaussianNB()
    # model = RandomForestClassifier(max_depth=2, random_state=0)
    # model = KNeighborsClassifier(n_neighbors=3)
    
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    # Acc = accuracy_score(y_test, y_pred)
    y_pred = torch.tensor( y_pred, dtype=torch.float32)
    
    # importances = model.feature_importances_

    accuracy = Accuracy(task="multiclass", num_classes=3)
    running_acc = accuracy(y_pred, y_test) 
    print('Testing acc={:.3f}'.format(running_acc))    
        
    f1s = MulticlassF1Score(num_classes=3)
    f1s = f1s(y_pred, y_test)
    print('F1 score={:.3f}'.format(f1s))
        
    recall = Recall(task="multiclass", average='macro', num_classes=3)
    recall = recall(y_pred, y_test)
    print('Recall={:.3f}'.format(recall))
        
    # auroc = AUROC(task="multiclass", num_classes=3)
    # auroc = auroc(y_pred, y_test)
    # print('AUROC={:.3f}'.format(auroc))
    

    # save results 
    


if __name__ == "__main__":
    main()