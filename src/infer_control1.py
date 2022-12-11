import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, roc_curve

import pickle

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BCFdata(Dataset):
    def __init__(self, pro_input, lig_input, bind_label):
        super(BCFdata, self).__init__()

        self.pro = pro_input
        self.lig = lig_input
        self.label = bind_label

    def __getitem__(self, index):
        item_pro = torch.tensor(self.pro[index], dtype = torch.float)
        item_lig = torch.tensor(self.lig[index], dtype = torch.float)
        item_label = torch.tensor(self.label[index], dtype = torch.float)
        
        item_pro = item_pro.unsqueeze(0)
        item_lig = item_lig.unsqueeze(0)
        item_label = item_label.unsqueeze(0).unsqueeze(0)
        
        return item_pro, item_lig, item_label


    def __len__(self):
        return len(self.label)



class BindingCF(nn.Module):

    def __init__(self):
        super(BindingCF, self).__init__()
        self.pconv1_1 = nn.Conv1d(1, 27, 45, padding=22) 
        self.pconv1_2 = nn.Conv1d(27, 81, 45, padding=22)
        self.pconv1_3 = nn.Conv1d(81, 243, 45, padding=22)
        self.pconv1_4 = nn.Conv1d(243, 1, 9, padding=4)
        
        
        self.lconv1_1 = nn.Conv1d(1, 9, 15, padding=7)
        self.lconv1_2 = nn.Conv1d(9, 27, 15, padding=7)
        self.lconv1_3 = nn.Conv1d(27, 81, 15, padding=7)
        self.lconv1_4 = nn.Conv1d(81, 1, 3, padding=1)
       
        self.maxpool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(.3) 
        self.fc1 = nn.Linear(80, 54)
        self.fc2 = nn.Linear(54, 27)
        self.fc3 = nn.Linear(27,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_pro, x_lig):
        x_pro1 = self.maxpool(F.relu(self.pconv1_1(x_pro)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_2(x_pro1)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_3(x_pro1)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_4(x_pro1)))
        
       

        x_lig1 = self.maxpool(F.relu(self.lconv1_1(x_lig)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_2(x_lig1)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_3(x_lig1)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_4(x_lig1)))
        



        x = torch.cat((x_pro1, x_lig1), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x



PATH_model = '/home/jy2/practice/pytorch/model_save/model_control1.pt'
PATH_valdata = '/home/jy2/practice/pytorch/model_save/data_val_control1.pt'

binding_cf = BindingCF()
binding_cf = nn.DataParallel(binding_cf)
binding_cf.to(device)
binding_cf.load_state_dict(torch.load(PATH_model), strict = True)
binding_cf.eval()

validationloader = torch.load(PATH_valdata)

label_true = []
label_score = []

correct = 0
with torch.no_grad():
    test_bind = 0
    test_non = 0
    num_of_TT = 0
    num_of_FF = 0
    num_of_TF = 0
    num_of_FT = 0

    for i, (pro_val, lig_val, label_val) in enumerate(validationloader):
        if label_val == 1:
            test_bind += 1
        else:
            test_non += 1

        pro_val = pro_val.to(device)
        lig_val = lig_val.to(device)
        label_val = label_val.to(device)

        preds = binding_cf(pro_val, lig_val)
        outputs = round(preds.item())
        label_score.append(preds.item())



        if outputs == label_val:
            correct += 1
        if outputs == 1 and label_val == 1:
            num_of_TT += 1
            label_true.append(1)
        elif outputs == 0 and label_val == 0:
            num_of_FF += 1
            label_true.append(0)
        elif outputs == 1 and label_val == 0:
            num_of_TF += 1
            label_true.append(0)
        elif outputs == 0 and label_val == 1:
            num_of_FT += 1
            label_true.append(1)



        

    print('TT = %d/FF = %d/TF = %d/FT = %d'%(num_of_TT,num_of_FF,num_of_TF,num_of_FT))
    print('#act_val : %d'%test_bind)
    print('#ina_val : %d'%test_non)
    print('#total : %d'%(test_bind+test_non))

print('correct : %d'%correct)
print('Accuracy : %d%%'%(100*correct/len(validationloader)))


with open("roc_label_true_control1.txt", "wb") as path_true:
    pickle.dump(label_true, path_true)
with open("roc_label_score_control1.txt", "wb") as path_score:
    pickle.dump(label_score, path_score)


