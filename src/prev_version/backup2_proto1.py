import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import datahelper
from datahelper import *

import random

os.environ["CUDA_VISIBLE_DEVICES"] ="0,1,2,3" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available)


class BindingCF(nn.Module):

    def __init__(self):
        super(BindingCF, self).__init__()
        
        self.pconv1 = nn.Conv1d(1, 1, 3) 
        self.pconv2 = nn.Conv1d(1, 1, 6) 
        self.lconv1 = nn.Conv1d(1, 1, 3)
        self.lconv2 = nn.Conv1d(1, 1, 4)
        self.maxpool = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(.3) 
        self.fc1 = nn.Linear(190, 90)
        self.fc2 = nn.Linear(90, 10)
        self.fc3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_lig, x_pro):
        x_pro = self.maxpool(F.relu(self.pconv1(x_pro)))
        x_pro = self.maxpool(F.relu(self.pconv2(x_pro)))

        x_lig = self.maxpool(F.relu(self.lconv1(x_lig)))
        x_lig = self.maxpool(F.relu(self.lconv2(x_lig)))

        x = torch.cat((x_pro, x_lig), -1)
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

trainloader = train_set
testloader = test_set

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        labels = torch.tensor([labels], dtype = torch.float).to(device)
        labels = labels.unsqueeze(0).unsqueeze(0)
        x_pro, x_lig = inputs
        x_pro = torch.tensor(x_pro, dtype = torch.float).to(device)
        x_lig = torch.tensor(x_lig, dtype = torch.float).to(device)
        x_pro = x_pro.unsqueeze(0).unsqueeze(0)
        x_lig = x_lig.unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()

        outputs = binding_cf(x_pro, x_lig)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'%(epoch + 1, i +1, running_loss/2000))
            running_loss = 0.0
        elif i == len(trainloader):
            print('[%d, %d] loss: %.3f'%(epoch + 1, num_of_train, running_loss/(num_of_train%2000)))
            running_loss = 0.0
print('Finished Training')

correct = 0
total = len(testloader)

with torch.no_grad():
    test_bind = 0
    test_non = 0
    for data in testloader:
        inputs, labels = data
        
        if labels == 0:
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
    print('bind : %s'%test_bind)
    print('inactive : %s'%test_non)
    print('%d%%'%(100*test_bind/total))
print('Accuracy : %d %%'%(100*correct/total))



