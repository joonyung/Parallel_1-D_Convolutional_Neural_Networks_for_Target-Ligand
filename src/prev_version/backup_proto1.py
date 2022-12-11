import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import datahelper
from datahelper import *


class BindingCF(nn.Module):

    def __init__(self):
        super(BindingCF, self).__init__()
        
        self.pconv1 = nn.Conv1d(1, 1, 3) 
        self.pconv2 = nn.Conv1d(1, 1, 6) 
        self.lconv1 = nn.Conv1d(1, 1, 3)
        self.lconv2 = nn.Conv1d(1, 1, 4)
        self.maxpool = nn.MaxPool1d(2)

#        self.dropout = nn.Dropout(.3) 
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
#        x = self.dropout(x)
        x = F.relu(self.fc2(x))
#        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


binding_cf = BindingCF()

criterion = nn.BCELoss()
optimizer = optim.SGD(binding_cf.parameters(), lr=0.0001, momentum=0.9)

trainloader = data_set

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        labels = torch.tensor([labels], dtype = torch.float)
        labels = labels.unsqueeze(0).unsqueeze(0)
        x_pro, x_lig = inputs
        x_pro = torch.tensor(x_pro, dtype = torch.float)
        x_lig = torch.tensor(x_lig, dtype = torch.float)
        x_pro = x_pro.unsqueeze(0).unsqueeze(0)
        x_lig = x_lig.unsqueeze(0).unsqueeze(0)
        optimizer.zero_grad()

        outputs = binding_cf(x_pro, x_lig)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f'%(epoch + 1, i +1, running_loss/200))
            running_loss = 0.0

print('Finished Training')
        
