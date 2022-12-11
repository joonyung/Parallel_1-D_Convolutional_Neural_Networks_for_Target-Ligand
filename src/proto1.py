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

BATCH_SIZE = 10
MAX_THREAD = 4
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
        
        self.pconv1_1 = nn.Conv1d(1, 9, 9) 
        self.pconv1_2 = nn.Conv1d(1, 27, 3)
        self.pconv2_1 = nn.Conv1d(1, 9, 27)
        self.pconv2_2 = nn.Conv1d(1, 27, 9)
        self.pconv3_1 = nn.Conv1d(1, 9, 81)
        self.pconv3_2 = nn.Conv1d(1, 27, 27)
        self.pconv4_1 = nn.Conv1d(1, 9, 243)
        self.pconv4_2 = nn.Conv1d(1, 27, 81)

        self.lconv1_1 = nn.Conv1d(1, 9, 9)
        self.lconv1_2 = nn.Conv1d(1, 27, 3)
        self.lconv2_1 = nn.Conv1d(1, 9, 27)
        self.lconv2_2 = nn.Conv1d(1, 27, 9)
        self.lconv3_1 = nn.Conv1d(1, 9, 81)
        self.lconv3_2 = nn.Conv1d(1, 27, 27)
        self.lconv4_1 = nn.Conv1d(1, 9, 243)
        self.lconv4_2 = nn.Conv1d(1, 27, 81)

        self.maxpool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(.3) 
        self.fc1 = nn.Linear(2444, 729)
        self.fc2 = nn.Linear(729, 81)
        self.fc3 = nn.Linear(81, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_pro, x_lig):
        x_pro1 = self.maxpool(F.relu(self.pconv1_1(x_pro)))
        x_pro1 = self.maxpool(F.relu(self.pconv1_2(x_pro)))

        x_pro2 = self.maxpool(F.relu(self.pconv2_1(x_pro)))
        x_pro2 = self.maxpool(F.relu(self.pconv2_2(x_pro)))

        x_pro3 = self.maxpool(F.relu(self.pconv3_1(x_pro)))
        x_pro3 = self.maxpool(F.relu(self.pconv3_2(x_pro)))

        x_pro4 = self.maxpool(F.relu(self.pconv4_1(x_pro)))
        x_pro4 = self.maxpool(F.relu(self.pconv4_2(x_pro)))

        x_lig1 = self.maxpool(F.relu(self.lconv1_1(x_lig)))
        x_lig1 = self.maxpool(F.relu(self.lconv1_2(x_lig)))

        x_lig2 = self.maxpool(F.relu(self.lconv2_1(x_lig)))
        x_lig2 = self.maxpool(F.relu(self.lconv2_2(x_lig)))

        x_lig3 = self.maxpool(F.relu(self.lconv3_1(x_lig)))
        x_lig3 = self.maxpool(F.relu(self.lconv3_2(x_lig)))

        x_lig4 = self.maxpool(F.relu(self.lconv4_1(x_lig)))
        x_lig4 = self.maxpool(F.relu(self.lconv4_2(x_lig)))

        x1 = torch.cat((x_pro1, x_lig1), -1)
        x2 = torch.cat((x_pro2, x_lig2), -1)
        x3 = torch.cat((x_pro3, x_lig3), -1)
        x4 = torch.cat((x_pro4, x_lig4), -1)

        x_1 = torch.cat((x1, x2), -1)
        x_2 = torch.cat((x3, x4), -1)
        x = torch.cat((x_1,x_2), -1)

        x = F.relu(self.fc1(x))
#        x = self.dropout(x)
        x = F.relu(self.fc2(x))
#        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


binding_cf = BindingCF()

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    binding_cf = nn.DataParallel(binding_cf)

binding_cf = binding_cf.to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(binding_cf.parameters(), lr=0.01, momentum=0.9)


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

trainloader = DataLoader(train_dataset, batch_size = 1024, shuffle = True)
validationloader = DataLoader(validation_dataset, batch_size = 1, shuffle =True)


#validation_data_saving
torch.save(validationloader, '/home/jy2/practice/pytorch/model_save/data_val_6.pt')

torch.save(BCFdata, '/home/jy2/practice/pytorch/model_save/data_model_BCFdata.pt')

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
PATH = '/home/jy2/practice/pytorch/model_save/model6.pt'
torch.save(binding_cf.state_dict(), PATH)



#validation
'''
correct = 0
with torch.no_grad():
    test_bind = 0
    test_non = 0
    
    for i, (pro_val, lig_val, label_val) in enumerate(validationloader):
        if label_val == 1:
            test_bind += 1
        else:
            test_non += 1

        pro_val = pro_val.to(device)
        lig_val = lig_val.to(device)
        label_val = label_val.to(device)

        outputs = binding_cf(pro_val, lig_val).round()
#        print(outputs, label_val)
        if outputs == label_val:
            correct += 1
    
    print('#act_val : %d'%test_bind)
    print('#ina_val : %d'%test_non)
    print('#total : %d'%(test_bind+test_non))
    print('of act_val : %d'%(100*test_bind/len(validationloader)))
print('correct : %d'%correct)
print('Accuracy : %d%%'%(100*correct/len(validationloader)))
'''











'''    
    for data in testloader:
        inputs, labels = data
        
        if labels == 1:
            test_bind += 1
        else:
            test_non += 1

        labels = torch.tensor([labels], dtype = torch.float).to(device)
#        print(labels)        
        x_pro, x_lig = inputs
        x_pro = torch.tensor(x_pro, dtype = torch.float).to(device)
        x_lig = torch.tensor(x_lig, dtype = torch.float).to(device)
        x_pro = x_pro.unsqueeze(0).unsqueeze(0)
        x_lig = x_lig.unsqueeze(0).unsqueeze(0)
        
        outputs = binding_cf(x_pro, x_lig)
#        _, predicted = torch.max(outputs.data,1)

        predicted = outputs.squeeze().round()
#        print(outputs)
        if predicted == labels:
            correct += 1
#            print(correct)
    print('num of act_test : %s'%test_bind)
    print('num of ina_test : %s'%test_non)
#    print('%d%%'%(100*test_bind/total))
print('Accuracy : %d %%'%(100*correct/total))
'''

'''
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'%(epoch + 1, i +1, running_loss/2000))
            running_loss = 0.0
        elif i == len(trainloader):
            print('[%d, %d] loss: %.3f'%(epoch + 1, num_of_train, running_loss/(num_of_train%2000)))
'''
 
'''
#from time import time
for epoch in range(2):
    running_loss = 0.0
#    tttt=[time()]
    for i, data in enumerate(trainloader, 0):
#        torch.cuda.synchronize()
#        tttt.append(time())
        inputs, labels = data
        labels = torch.tensor([labels], dtype = torch.float).to(device)
        labels = labels.unsqueeze(0).unsqueeze(0)
        x_pro, x_lig = inputs
        x_pro = torch.tensor(x_pro, dtype = torch.float).to(device)
        x_lig = torch.tensor(x_lig, dtype = torch.float).to(device)
        x_pro = x_pro.unsqueeze(0).unsqueeze(0)
        x_lig = x_lig.unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()

#        torch.cuda.synchronize()
#        tttt.append(time())
        outputs = binding_cf(x_pro, x_lig)
        loss = criterion(outputs, labels)
        loss.backward()
#        torch.cuda.synchronize()
#        tttt.append(time())
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'%(epoch + 1, i +1, running_loss/2000))
            running_loss = 0.0
        elif i == len(trainloader):
            print('[%d, %d] loss: %.3f'%(epoch + 1, num_of_train, running_loss/(num_of_train%2000)))
            running_loss = 0.0
#        torch.cuda.synchronize()
#        tttt.append(time())
#        tttt = np.array(tttt,dtype=np.float)
#        print('my_time',tttt[1:]-tttt[:-1])
#        tttt = [time()]
print('Finished Training')
'''
2

''' 
VALIDATION
correct = 0
total = len(testloader)

with torch.no_grad():
    test_bind = 0
    test_non = 0
    for data in testloader:
        inputs, labels = data
        
        if labels == 1:
            test_bind += 1
#tttt=list()
        else:
            test_non += 1

        labels = torch.tensor([labels], dtype = torch.float).to(device)
#        print(labels)        
        x_pro, x_lig = inputs
        x_pro = torch.tensor(x_pro, dtype = torch.float).to(device)
        x_lig = torch.tensor(x_lig, dtype = torch.float).to(device)
        x_pro = x_pro.unsqueeze(0).unsqueeze(0)
        x_lig = x_lig.unsqueeze(0).unsqueeze(0)
        
        outputs = binding_cf(x_pro, x_lig)
#        _, predicted = torch.max(outputs.data,1)

        predicted = outputs.squeeze().round()
#        print(outputs)
        if predicted == labels:
            correct += 1
#            print(correct)
    print('num of act_test : %s'%test_bind)
    print('num of ina_test : %s'%test_non)
#    print('%d%%'%(100*test_bind/total))
print('Accuracy : %d %%'%(100*correct/total))
'''
