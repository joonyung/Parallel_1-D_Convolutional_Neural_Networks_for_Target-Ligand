import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

import datahelper
from datahelper import *

import random

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#BATCH_SIZE = 10
#MAX_THREAD = 4
#TR_LOADER_PARAMS = {'batch_size' : BATCH_SIZE,  'shuffle': True, 'num_workers' : MAX_THREAD, 'pin_memory' : True}
#VAL_LOADER_PARAMS = {'batch_size' : BATCH_SIZE,  'shuffle': False, 'num_workers' : MAX_THREAD, 'pin_memory' : True}


print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available)


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


binding_cf = BindingCF()

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    binding_cf = nn.DataParallel(binding_cf)

binding_cf = binding_cf.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(binding_cf.parameters(), lr=0.001, momentum=0.9)


go_set_pro = []
go_set_lig = []
go_set_label = []


for i, (data_pro, data_lig, data_label) in enumerate(data_set):
    go_set_pro.append(data_pro)
    go_set_lig.append(data_lig)
    go_set_label.append(data_label)

total_set = BCFdata(go_set_pro, go_set_lig, go_set_label)

#train/val division
num_of_val = len(total_set)//6
train_dataset, validation_dataset = random_split(total_set, [len(total_set) - num_of_val, num_of_val])

trainloader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
validationloader = DataLoader(validation_dataset, batch_size = 1, shuffle =True)


#validation_data_saving
torch.save(validationloader, '/home/jy2/practice/pytorch/model_save/data_val_control.pt')

for epoch in range(50):
    running_loss = 0.0

    print("#train data per batch : %d"%len(trainloader))    
    for i, (pro_train, lig_train, label_train) in enumerate(trainloader):

        pro_train = pro_train.to(device)
        lig_train = lig_train.to(device)
        label_train = label_train.to(device)
        
        optimizer.zero_grad()

        outputs = binding_cf(pro_train, lig_train)
        loss = criterion(outputs, label_train)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print("loss in epoch %d : %.10f"%(epoch+1, running_loss/len(trainloader)))
        
print('Finished Training')

#saving model
PATH = '/home/jy2/practice/pytorch/model_save/model_control.pt'
torch.save(binding_cf.state_dict(), PATH)



