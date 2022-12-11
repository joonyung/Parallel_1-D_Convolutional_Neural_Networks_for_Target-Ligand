import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os


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

        self.pconv1_1 = nn.Conv1d(1, 27, 9) 
        self.pconv1_2 = nn.Conv1d(27, 27, 9)
        self.pconv1_3 = nn.Conv1d(27, 1, 9)
        self.pconv2_1 = nn.Conv1d(1, 27, 27)
        self.pconv2_2 = nn.Conv1d(27, 27, 27)
        self.pconv2_3 = nn.Conv1d(27, 1, 9)
        self.pconv3_1 = nn.Conv1d(1, 27, 81)
        self.pconv3_2 = nn.Conv1d(27, 27, 81)
        self.pconv3_3 = nn.Conv1d(27, 1, 9)
        self.pconv4_1 = nn.Conv1d(1, 27, 243)
        self.pconv4_2 = nn.Conv1d(27, 27, 243)
        self.pconv4_3 = nn.Conv1d(27, 1, 9)

        self.lconv1_1 = nn.Conv1d(1, 9, 3)
        self.lconv1_2 = nn.Conv1d(9, 9, 3)
        self.lconv1_3 = nn.Conv1d(9, 1, 3)
        self.lconv2_1 = nn.Conv1d(1, 9, 9)
        self.lconv2_2 = nn.Conv1d(9, 9, 9)
        self.lconv2_3 = nn.Conv1d(9, 1, 3)
        self.lconv3_1 = nn.Conv1d(1, 9, 27)
        self.lconv3_2 = nn.Conv1d(9, 9, 27)
        self.lconv3_3 = nn.Conv1d(9, 1, 3)
        self.lconv4_1 = nn.Conv1d(1, 9, 81)
        self.lconv4_2 = nn.Conv1d(9, 9, 81)
        self.lconv4_3 = nn.Conv1d(9, 1, 3)

        self.maxpool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(.3) 
        self.fc1 = nn.Linear(442, 243)
        self.fc2 = nn.Linear(243, 81)
        self.fc3 = nn.Linear(81,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_pro, x_lig):
        x_pro1 = self.maxpool(F.relu(self.pconv1_1(x_pro)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_2(x_pro1)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_3(x_pro1)))

        x_pro2 = self.maxpool(F.relu(self.pconv2_1(x_pro)))
        x_pro2 = self.maxpool(F.relu(self.pconv2_2(x_pro2)))
        x_pro2 = self.maxpool(F.relu(self.pconv2_3(x_pro2)))

        x_pro3 = self.maxpool(F.relu(self.pconv3_1(x_pro)))
        x_pro3 = self.maxpool(F.relu(self.pconv3_2(x_pro3)))
        x_pro3 = self.maxpool(F.relu(self.pconv3_3(x_pro3)))

        x_pro4 = self.maxpool(F.relu(self.pconv4_1(x_pro)))
        x_pro4 = self.maxpool(F.relu(self.pconv4_2(x_pro4)))
        x_pro4 = self.maxpool(F.relu(self.pconv4_3(x_pro4)))       
        
        x_lig1 = self.maxpool(F.relu(self.lconv1_1(x_lig)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_2(x_lig1)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_3(x_lig1)))

        x_lig2 = self.maxpool(F.relu(self.lconv2_1(x_lig)))
        x_lig2 = self.maxpool(F.relu(self.lconv2_2(x_lig2)))
        x_lig2 = self.maxpool(F.relu(self.lconv2_3(x_lig2)))

        x_lig3 = self.maxpool(F.relu(self.lconv3_1(x_lig)))
        x_lig3 = self.maxpool(F.relu(self.lconv3_2(x_lig3)))
        x_lig3 = self.maxpool(F.relu(self.lconv3_3(x_lig3)))

        x_lig4 = self.maxpool(F.relu(self.lconv4_1(x_lig)))
        x_lig4 = self.maxpool(F.relu(self.lconv4_2(x_lig4)))
        x_lig4 = self.maxpool(F.relu(self.lconv4_3(x_lig4)))


        x1 = torch.cat((x_pro1, x_lig1), -1)
        x2 = torch.cat((x_pro2, x_lig2), -1)
        x3 = torch.cat((x_pro3, x_lig3), -1)
        x4 = torch.cat((x_pro4, x_lig4), -1)

        x_1 = torch.cat((x1, x2), -1)
        x_2 = torch.cat((x3, x4), -1)
        x = torch.cat((x_1,x_2), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x



PATH_model = '/home/jy2/practice/pytorch/model_save/model8.pt'
PATH_valdata = '/home/jy2/practice/pytorch/model_save/data_val_8.pt'

binding_cf = BindingCF()
binding_cf = nn.DataParallel(binding_cf)
binding_cf.to(device)
binding_cf.load_state_dict(torch.load(PATH_model), strict = True)
binding_cf.eval()

validationloader = torch.load(PATH_valdata)

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
        elif outputs == 0 and label_val == 0:
            num_of_FF += 1
        elif outputs == 1 and label_val == 0:
            num_of_TF += 1
        elif outputs == 0 and label_val == 1:
            num_of_FT += 1



    print('TT = %d/FF = %d/TF = %d/FT = %d'%(num_of_TT,num_of_FF,num_of_TF,num_of_FT))
    print('#act_val : %d'%test_bind)
    print('#ina_val : %d'%test_non)
    print('#total : %d'%(test_bind+test_non))

print('correct : %d'%correct)
print('Accuracy : %d%%'%(100*correct/len(validationloader)))
 


