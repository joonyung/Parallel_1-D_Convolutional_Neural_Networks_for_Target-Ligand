import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, roc_curve
import matplotlib
matplotlib.use('Agg')


os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 10
MAX_THREAD = 4
#TR_LOADER_PARAMS = {'batch_size' : BATCH_SIZE,  'shuffle': True, 'num_workers' : MAX_THREAD, 'pin_memory' : True}
#VAL_LOADER_PARAMS = {'batch_size' : BATCH_SIZE,  'shuffle': False, 'num_workers' : MAX_THREAD, 'pin_memory' : True}
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

        self.pconv1_1 = nn.Conv1d(1, 27, 9, padding=4) 
        self.pconv1_2 = nn.Conv1d(27, 81, 9, padding=4)
        self.pconv1_3 = nn.Conv1d(81, 243, 9, padding=4)
        self.pconv1_4 = nn.Conv1d(243, 1, 9, padding=4)
        
        self.pconv2_1 = nn.Conv1d(1, 27, 15, padding=7) 
        self.pconv2_2 = nn.Conv1d(27, 81, 15, padding=7)
        self.pconv2_3 = nn.Conv1d(81, 243, 15, padding=7)
        self.pconv2_4 = nn.Conv1d(243, 1, 9, padding=4)

        self.pconv3_1 = nn.Conv1d(1, 27, 27, padding=13)
        self.pconv3_2 = nn.Conv1d(27, 81, 27, padding=13)
        self.pconv3_3 = nn.Conv1d(81, 243, 27, padding=13)
        self.pconv3_4 = nn.Conv1d(243, 1, 9, padding=4)
        
        self.pconv4_1 = nn.Conv1d(1, 27, 45, padding=22) 
        self.pconv4_2 = nn.Conv1d(27, 81, 45, padding=22)
        self.pconv4_3 = nn.Conv1d(81, 243, 45, padding=22)
        self.pconv4_4 = nn.Conv1d(243, 1, 9, padding=4)

        self.pconv5_1 = nn.Conv1d(1, 27, 81, padding=40)
        self.pconv5_2 = nn.Conv1d(27, 81, 81, padding=40)
        self.pconv5_3 = nn.Conv1d(81, 243, 81, padding=40)
        self.pconv5_4 = nn.Conv1d(243, 1, 9, padding=4)
        
        self.pconv6_1 = nn.Conv1d(1, 27, 135, padding=67)
        self.pconv6_2 = nn.Conv1d(27, 81, 135, padding=67)
        self.pconv6_3 = nn.Conv1d(81, 243, 135, padding=67)
        self.pconv6_4 = nn.Conv1d(243, 1, 9, padding=4)

        self.pconv7_1 = nn.Conv1d(1, 27, 243, padding=121)
        self.pconv7_2 = nn.Conv1d(27, 81, 243, padding=121)
        self.pconv7_3 = nn.Conv1d(81, 243, 243, padding=121)
        self.pconv7_4 = nn.Conv1d(243, 1, 9, padding=4)
        
        self.pconv8_1 = nn.Conv1d(1, 27, 405, padding = 202)
        self.pconv8_2 = nn.Conv1d(27, 81, 405, padding = 202)
        self.pconv8_3 = nn.Conv1d(81, 243, 405, padding = 202)
        self.pconv8_4 = nn.Conv1d(243, 1, 9, padding=4)
 

        self.lconv1_1 = nn.Conv1d(1, 9, 3, padding=1)
        self.lconv1_2 = nn.Conv1d(9, 27, 3, padding=1)
        self.lconv1_3 = nn.Conv1d(27, 81, 3, padding=1)
        self.lconv1_4 = nn.Conv1d(81, 1, 3, padding=1)
       
        self.lconv2_1 = nn.Conv1d(1, 9, 5, padding=2)
        self.lconv2_2 = nn.Conv1d(9, 27, 5, padding=2)
        self.lconv2_3 = nn.Conv1d(27, 81, 5, padding=2)
        self.lconv2_4 = nn.Conv1d(81, 1, 5, padding=1)  
        
        self.lconv3_1 = nn.Conv1d(1, 9, 9, padding=4)
        self.lconv3_2 = nn.Conv1d(9, 27, 9, padding=4)
        self.lconv3_3 = nn.Conv1d(27, 81, 9, padding=4)
        self.lconv3_4 = nn.Conv1d(81, 1, 3, padding=1)
        
        self.lconv4_1 = nn.Conv1d(1, 9, 15, padding=7)
        self.lconv4_2 = nn.Conv1d(9, 27, 15, padding=7)
        self.lconv4_3 = nn.Conv1d(27, 81, 15, padding=7)
        self.lconv4_4 = nn.Conv1d(81, 1, 3, padding=1)

        self.lconv5_1 = nn.Conv1d(1, 9, 27, padding=13)
        self.lconv5_2 = nn.Conv1d(9, 27, 27, padding=13)
        self.lconv5_3 = nn.Conv1d(27, 81, 27, padding=13)
        self.lconv5_4 = nn.Conv1d(81, 1, 3, padding=1)
 
        self.lconv6_1 = nn.Conv1d(1, 9, 45, padding=22)
        self.lconv6_2 = nn.Conv1d(9, 27, 45, padding=22)
        self.lconv6_3 = nn.Conv1d(27, 81, 45, padding=22)
        self.lconv6_4 = nn.Conv1d(81, 1, 3, padding=1)
        
        self.lconv7_1 = nn.Conv1d(1, 9, 81, padding =40)
        self.lconv7_2 = nn.Conv1d(9, 27, 81, padding =40)
        self.lconv7_3 = nn.Conv1d(27, 81, 81, padding =40)
        self.lconv7_4 = nn.Conv1d(81, 1, 3, padding=1)
        
        self.lconv8_1 = nn.Conv1d(1, 9, 135, padding=67)
        self.lconv8_2 = nn.Conv1d(9, 27, 135, padding=67)
        self.lconv8_3 = nn.Conv1d(27, 81, 135, padding=67)
        self.lconv8_4 = nn.Conv1d(81, 1, 3, padding=1)
 

        self.maxpool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(.3) 
        self.fc1 = nn.Linear(639, 243)
        self.fc2 = nn.Linear(243, 81)
        self.fc3 = nn.Linear(81,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_pro, x_lig):

        x_pro1 = self.maxpool(F.relu(self.pconv1_1(x_pro)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_2(x_pro1)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_3(x_pro1)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_4(x_pro1)))
        
        x_pro2 = self.maxpool(F.relu(self.pconv2_1(x_pro)))
        x_pro2 = self.maxpool(F.relu(self.pconv2_2(x_pro2)))
        x_pro2 = self.maxpool(F.relu(self.pconv2_3(x_pro2)))
        x_pro2 = self.maxpool(F.relu(self.pconv2_4(x_pro2)))
        
        x_pro3 = self.maxpool(F.relu(self.pconv3_1(x_pro)))
        x_pro3 = self.maxpool(F.relu(self.pconv3_2(x_pro3)))
        x_pro3 = self.maxpool(F.relu(self.pconv3_3(x_pro3)))
        x_pro3 = self.maxpool(F.relu(self.pconv3_4(x_pro3)))
        
        x_pro4 = self.maxpool(F.relu(self.pconv4_1(x_pro)))
        x_pro4 = self.maxpool(F.relu(self.pconv4_2(x_pro4)))
        x_pro4 = self.maxpool(F.relu(self.pconv4_3(x_pro4)))
        x_pro4 = self.maxpool(F.relu(self.pconv4_4(x_pro4)))
        
        x_pro5 = self.maxpool(F.relu(self.pconv5_1(x_pro)))
        x_pro5 = self.maxpool(F.relu(self.pconv5_2(x_pro5)))
        x_pro5 = self.maxpool(F.relu(self.pconv5_3(x_pro5)))
        x_pro5 = self.maxpool(F.relu(self.pconv5_4(x_pro5)))
        
        x_pro6 = self.maxpool(F.relu(self.pconv6_1(x_pro)))
        x_pro6 = self.maxpool(F.relu(self.pconv6_2(x_pro6)))
        x_pro6 = self.maxpool(F.relu(self.pconv6_3(x_pro6)))
        x_pro6 = self.maxpool(F.relu(self.pconv6_4(x_pro6)))

        x_pro7 = self.maxpool(F.relu(self.pconv7_1(x_pro)))
        x_pro7 = self.maxpool(F.relu(self.pconv7_2(x_pro7)))
        x_pro7 = self.maxpool(F.relu(self.pconv7_3(x_pro7)))
        x_pro7 = self.maxpool(F.relu(self.pconv7_4(x_pro7)))

        x_pro8 = self.maxpool(F.relu(self.pconv8_1(x_pro)))
        x_pro8 = self.maxpool(F.relu(self.pconv8_2(x_pro8)))
        x_pro8 = self.maxpool(F.relu(self.pconv8_3(x_pro8)))
        x_pro8 = self.maxpool(F.relu(self.pconv8_4(x_pro8)))
        

        x_lig1 = self.maxpool(F.relu(self.lconv1_1(x_lig)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_2(x_lig1)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_3(x_lig1)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_4(x_lig1)))
        
        x_lig2 = self.maxpool(F.relu(self.lconv2_1(x_lig)))
        x_lig2 = self.maxpool(F.relu(self.lconv2_2(x_lig2)))
        x_lig2 = self.maxpool(F.relu(self.lconv2_3(x_lig2)))
        x_lig2 = self.maxpool(F.relu(self.lconv2_4(x_lig2)))
        
        x_lig3 = self.maxpool(F.relu(self.lconv3_1(x_lig)))
        x_lig3 = self.maxpool(F.relu(self.lconv3_2(x_lig3)))
        x_lig3 = self.maxpool(F.relu(self.lconv3_3(x_lig3)))
        x_lig3 = self.maxpool(F.relu(self.lconv3_4(x_lig3)))
        
        x_lig4 = self.maxpool(F.relu(self.lconv4_1(x_lig)))
        x_lig4 = self.maxpool(F.relu(self.lconv4_2(x_lig4)))
        x_lig4 = self.maxpool(F.relu(self.lconv4_3(x_lig4)))
        x_lig4 = self.maxpool(F.relu(self.lconv4_4(x_lig4)))
        
        x_lig5 = self.maxpool(F.relu(self.lconv5_1(x_lig)))
        x_lig5 = self.maxpool(F.relu(self.lconv5_2(x_lig5)))
        x_lig5 = self.maxpool(F.relu(self.lconv5_3(x_lig5)))
        x_lig5 = self.maxpool(F.relu(self.lconv5_4(x_lig5)))
        
        x_lig6 = self.maxpool(F.relu(self.lconv6_1(x_lig)))
        x_lig6 = self.maxpool(F.relu(self.lconv6_2(x_lig6)))
        x_lig6 = self.maxpool(F.relu(self.lconv6_3(x_lig6)))
        x_lig6 = self.maxpool(F.relu(self.lconv6_4(x_lig6)))

        x_lig7 = self.maxpool(F.relu(self.lconv7_1(x_lig)))
        x_lig7 = self.maxpool(F.relu(self.lconv7_2(x_lig7)))
        x_lig7 = self.maxpool(F.relu(self.lconv7_3(x_lig7)))
        x_lig7 = self.maxpool(F.relu(self.lconv7_4(x_lig7)))

        x_lig8 = self.maxpool(F.relu(self.lconv8_1(x_lig)))
        x_lig8 = self.maxpool(F.relu(self.lconv8_2(x_lig8)))
        x_lig8 = self.maxpool(F.relu(self.lconv8_3(x_lig8)))
        x_lig8 = self.maxpool(F.relu(self.lconv8_4(x_lig8)))


        x1 = torch.cat((x_pro1, x_lig1), -1)
        x2 = torch.cat((x_pro2, x_lig2), -1)
        x3 = torch.cat((x_pro3, x_lig3), -1)
        x4 = torch.cat((x_pro4, x_lig4), -1)
        x5 = torch.cat((x_pro5, x_lig5), -1)
        x6 = torch.cat((x_pro6, x_lig6), -1)
        x7 = torch.cat((x_pro7, x_lig7), -1)
        x8 = torch.cat((x_pro8, x_lig8), -1)


        x_1 = torch.cat((x1, x2), -1)
        x_2 = torch.cat((x3, x4), -1)
        x_3 = torch.cat((x5, x6), -1)
        x_4 = torch.cat((x7, x8), -1)

        x_a = torch.cat((x_1,x_2), -1)
        x_b = torch.cat((x_3,x_4), -1)

        x = torch.cat((x_a,x_b), -1)


        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x



PATH_model = '/home/jy2/practice/pytorch/model_save/model9.pt'
PATH_valdata = '/home/jy2/practice/pytorch/model_save/data_val_9.pt'

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

        outputs = binding_cf(pro_val, lig_val).round()
        
        if outputs == label_val:
            correct += 1
        if outputs == 1 and label_val == 1:
            num_of_TT += 1
            label_true.append(1)
            label_score.append(1)
        elif outputs == 0 and label_val == 0:
            num_of_FF += 1
            label_true.append(0)
            label_score.append(0)
        elif outputs == 1 and label_val == 0:
            num_of_TF += 1
            label_true.append(0)
            label_score.append(1)
        elif outputs == 0 and label_val == 1:
            num_of_FT += 1
            label_true.append(1)
            label_score.append(0)



    print('TT = %d/FF = %d/TF = %d/FT = %d'%(num_of_TT,num_of_FF,num_of_TF,num_of_FT))
    print('#act_val : %d'%test_bind)
    print('#ina_val : %d'%test_non)
    print('#total : %d'%(test_bind+test_non))

print('correct : %d'%correct)
print('Accuracy : %d%%'%(100*correct/len(validationloader)))
 
label_true = np.array(label_true)
label_score = np.array(label_score)

fpr, tpr, thresholds = roc_curve(label_true, label_score)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
print('0000')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('Flase Positive RAte')

plt.savefig('model9_1.png')
