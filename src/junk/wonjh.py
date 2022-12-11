#!/usr/bin/env python

import sys
argv = sys.argv
valid_mode = False
print_mode = False
if 'valid' in argv:
    valid_mode = True
    valid_fn_s = argv[argv.index('valid')+1:]
elif 'print' in argv:
    print_mode = True
    print_fn = argv[argv.index('print')+1]
#dev = "0"
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = dev
if "VALIDATE_IDX" in os.environ:
    valid_idx = int(os.environ["VALIDATE_IDX"])
else:
    valid_idx = 0
SUCCESS_CUTOFF = 2.0
    
def main():
    if valid_mode:
        valid(valid_fn_s)
    elif print_mode:
        print_weight(print_fn)
    else:
        train()
    return

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from time import time
from random import shuffle, sample
from glob import glob

#device = torch.device('cuda')
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_NEIGHBOR = 10
N_STR_SEL = 32#64

#STANDARD = 'NO_MORE_NEIGHBOR ALA GLY ILE LEU PRO VAL PHE TRP TYR ASP GLU ARG HIS LYS SER THR CYS MET ASN GLN'.split()
#AA_IDX = dict()
#for i,s in enumerate(STANDARD): AA_IDX[s] = i
#GLY_IDX = AA_IDX['GLY']
#BB_ATOM = ['CA','N ','C ','O ','CB'] # CB should be the last
#COORD_IDX = dict()
#for i,bb in enumerate(BB_ATOM): COORD_IDX[bb] = i
#CB_IDX = COORD_IDX['CB']

TRAIN_VAL_RATIO = 4
VALIDATE_INTERVAL = 1
BATCH_SIZE = 16
PRINT_TRAIN_STEP = BATCH_SIZE * 10000

MAX_THREAD = 8
if (not torch.cuda.is_available()) and ('NSLOTS' in os.environ):
    MAX_THREAD = int(os.environ['NSLOTS'])
TR_LOADER_PARAMS = {'batch_size': 1, 'shuffle': True, 'num_workers': MAX_THREAD, 'pin_memory':True}
VAL_LOADER_PARAMS = {'batch_size': 1, 'shuffle': False, 'num_workers': MAX_THREAD, 'pin_memory':True}
torch.set_num_threads(MAX_THREAD+2)

run_name = (sys.argv[0]).split('/')[-1][:-3] + '.%d'%valid_idx

def set_prep():
    train_s = list()
    val_s = list()
    i = 0
    if valid_mode:
        val_s = [l.strip() for l in open('valid.set')]
        return train_s, val_s
    with open('train.set') as f:
        for l in f:
            if i % TRAIN_VAL_RATIO != valid_idx:
                train_s.append(l.strip())
            else:
                val_s.append(l.strip())
            i+=1
    return train_s, val_s

class MyDataset(Dataset):
    def __init__(self, target_s, is_train=False):
        self.target_s = target_s
        self.is_train = is_train
    def __len__(self):
        return len(self.target_s)
    def __getitem__(self, index):
        tar = self.target_s[index]
        #
        # data loading
        # score
        rmsd = np.load('data/%s_rmsd.npy'%tar)
        if not valid_mode:
            rmsd = np.clip(rmsd,0.,3.999)
        # coordinates
        coord_r = np.load('data/%s_coord_r.npy'%tar)
        coord_l = np.load('data/%s_coord_l.npy'%tar)
        # atomtype / charge
        atom_r = np.load('data/%s_atomtype_r.npy'%tar)
        atom_l = np.load('data/%s_atomtype_l.npy'%tar)
        charge = np.load('data/%s_charge_l.npy'%tar)
        #
        n_model = rmsd.shape[0]
        # sample subset
        if self.is_train:
            # N_STR_SEL = 32
            # 0~1: max 32/4; 1~2: max 32/4; 2~3: max 32/4; 3~: rest
            category = rmsd.astype(np.int)
            sel = list()
            for star in (0,1,2):
                idx = np.where(category == star)[0]
                shuffle(idx)
                sel.extend(idx[:min(N_STR_SEL/4,(len(idx)+1)/2)])
            # rest: random incorrects
            idx = np.where(category == 3)[0]
            shuffle(idx)
            sel.extend(idx[:(N_STR_SEL-len(sel))])
            sel = np.array(sel, dtype=np.int)
            rmsd = rmsd[sel]
            coord_l = coord_l[sel]
            n_model = len(sel)
        #
        return tar, rmsd, coord_r, coord_l, atom_r, atom_l, charge, n_model


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #self.fc1 = nn.Linear(21, 8, bias=False)
        self.fc1_1 = nn.Embedding(25, 8)
        self.fc1_2 = nn.Embedding(25, 8)
        #
        self.fc2_0_1 = nn.Linear(9, 64)
        self.fc2_0_2 = nn.Linear(9, 64, bias=False)
        self.fc2_1 = nn.Linear(64, 64)
        self.fc2_2 = nn.Linear(64, 64)
        self.fc2_3 = nn.Linear(64, 64)
        self.n_hidden_2_4 = 64
        self.fc2_4 = nn.Linear(64, self.n_hidden_2_4)
        #
        self.fc3_0 = nn.Linear(self.n_hidden_2_4, 64)
        self.fc3_1 = nn.Linear(64, 64)
        self.fc3_2 = nn.Linear(64, 64)
        self.fc3_3 = nn.Linear(64, 64)
        self.fc3_4 = nn.Linear(64, 64)
        #
        self.fc4 = nn.Linear(64, 1, bias=False)
        #
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.5)
        #self.sigmoid = nn.Sigmoid()
        #
    def forward(self, x1, x2, x3, x4):
        atom_r_outputs = self.fc1_1(x1)
        atom_l_outputs = self.fc1_2(x3)
        #
        tmp1 = torch.cat([atom_r_outputs, x2.unsqueeze(-1)], dim=3)
        tmp2 = torch.cat([atom_l_outputs, x4.unsqueeze(-1)], dim=1)
        #
        #tmp1 = self.dropout2(tmp1)
        #tmp2 = self.dropout2(tmp2)
        tmp = self.fc2_0_1(tmp1) + self.fc2_0_2(tmp2).unsqueeze(0).unsqueeze(2)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc2_1(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc2_2(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc2_3(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc2_4(tmp)
        tmp = self.relu(tmp)
        tmp = torch.sum(tmp, dim=2)
        #tmp = torch.max(tmp, dim=0)[0]
        #
        tmp = self.fc3_0(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc3_1(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc3_2(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc3_3(tmp)
        tmp = self.relu(tmp)
        #tmp = self.dropout5(tmp)
        tmp = self.fc3_4(tmp)
        tmp = self.relu(tmp)
        #
        #y = self.sigmoid(self.fc4(tmp))
        y = self.fc4(tmp)
        return y

model = MyModel().to(device)

def corr(output, target):
    #output = output.view(-1)
    #target = target.view(-1)
    #
    epsilon = 1.e-10
    #
    mx = output.mean()
    x = output - mx
    sdx = (x**2).mean() + epsilon
    x = x / (sdx**0.5)
    #
    my = target.mean()
    y = target - my
    sdy = (y**2).mean() + epsilon
    y = y / (sdy**0.5)
    #
    loss = (x*y).mean()
    return loss

def distinguish_global(output, target):
    D = 0.5
    epsilon = 1.e-10
    #
    x1 = output.unsqueeze(1)
    x2 = output.unsqueeze(0)
    with torch.no_grad():
        y1 = target.unsqueeze(1)
        y2 = target.unsqueeze(0)
        #
        #compare = y1 - y2 - D + epsilon
        compare = - y1 + y2 - D + epsilon
        compare = compare.sign() / 2.0 + 0.5
        weight = compare / (compare.sum() + epsilon)
        #weight = compare / (compare.sum((1,2,3),keepdim=True) + epsilon)
    #
    loss = (x1 - x2 + 1.0).clamp(min=0.0)
    return (loss * weight).sum()

def top1_loss(output, target):
    idx = torch.argmin(output)
    return target[idx] - target.min()

def top5_loss(output, target):
    idx = torch.topk(output, 5, largest=False)[1]
    return target[idx].min() - target.min()

def top1_success(output, target):
    idx = torch.argmin(output)
    return 1.0 if target[idx] <= SUCCESS_CUTOFF else 0.0
    
def top5_success(output, target):
    idx = torch.topk(output, 5, largest=False)[1]
    return 1.0 if target[idx].min() <= SUCCESS_CUTOFF else 0.0

def generate_feature(atom_r, coord_r, coord_l):
    with torch.no_grad():
        n_model = coord_l.shape[0]
        # distances
        y1 = coord_l.unsqueeze(2)
        y2 = coord_r.unsqueeze(0).unsqueeze(1)
        d_s = ((y1 - y2)**2).sum(-1)
        #
        # receptor residues on interface
        d_close, neighbor = torch.topk(d_s, N_NEIGHBOR, dim=2, largest=False)
        #
        # amino acid features
        feature1 = atom_r[neighbor]
        # distance features
        feature2 = d_close
    return feature1, feature2

def train():
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    start_epoch = 1
    tmp_fn_model = 'model/%s_current.pt'%run_name
    if not os.path.exists('model'): os.mkdir('model')
    if os.path.exists(tmp_fn_model):
        checkpoint = torch.load(tmp_fn_model, map_location=device)
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    #scheduler.last_epoch = start_epoch
    
    train_s, val_s = set_prep()
    training_set = MyDataset(train_s,True)
    training_generator = DataLoader(training_set, **TR_LOADER_PARAMS)
    n_training_set = len(training_generator)
    
    validate_set = MyDataset(val_s,True)
    validate_generator = DataLoader(validate_set, **VAL_LOADER_PARAMS)
    
    for epoch in range(start_epoch,1001):
    #    scheduler.step()
        i = 0; sum_loss1 = 0.0; sum_local_loss1 = 0.0; sum_loss2 = 0.0; sum_local_loss2 = 0.0; sum_loss3 = 0.0; sum_local_loss3 = 0.0
        t0 = time()
        tmp = list()
        model.train()
        optimizer.zero_grad()
        for _, rmsd, coord_r, coord_l, atom_r, atom_l, charge, n_model in training_generator:
            coord_l = coord_l[0].to(device=device,non_blocking=True)
            coord_r = coord_r[0].to(device=device,non_blocking=True)
            rmsd = rmsd[0].to(device=device,non_blocking=True)
            atom_l = atom_l[0].to(device=device,non_blocking=True)
            atom_r = atom_r[0].to(device=device,non_blocking=True)
            charge = charge[0].to(device=device,non_blocking=True)
            #
            feature1, feature2 = generate_feature(atom_r, coord_r, coord_l)
            # feed forward
            energy_per_residue = model(feature1, feature2, atom_l, charge)
            # model-wise sum
            energy = energy_per_residue.sum(dim=1).view(-1)
            loss1 = distinguish_global(energy, rmsd)
            sum_loss1_target = loss1.item()
            loss2 = top1_success(energy, rmsd)
            sum_loss2_target = loss2#.item()
            loss3 = top5_success(energy, rmsd)
            sum_loss3_target = loss3#.item()
            loss1.backward()
            sum_local_loss1 += sum_loss1_target
            sum_local_loss2 += sum_loss2_target
            sum_local_loss3 += sum_loss3_target
            #
            i+=1
            if i % BATCH_SIZE == 0:
                optimizer.step()
                optimizer.zero_grad()
            if i % PRINT_TRAIN_STEP == 0:
                sum_loss1 += sum_local_loss1
                sum_loss2 += sum_local_loss2
                sum_loss3 += sum_local_loss3
                print '%4d/%d, loss: %.3f %.3f %.3f'%(i, n_training_set, sum_local_loss1/PRINT_TRAIN_STEP, sum_local_loss2/PRINT_TRAIN_STEP, sum_local_loss3/PRINT_TRAIN_STEP)
                sum_local_loss1 = 0.0; sum_local_loss2 = 0.0; sum_local_loss3 = 0.0
        if i % BATCH_SIZE > 0:
            optimizer.step()
        sum_loss1 += sum_local_loss1
        sum_loss2 += sum_local_loss2
        sum_loss3 += sum_local_loss3
        t1 = time()
        print 'tr, epoch: %3d loss: %.3f %.3f %.3f time: %.2f min'%(epoch, sum_loss1/i, sum_loss2/i, sum_loss3/i, (t1-t0)/60.)
    
        if os.path.exists(tmp_fn_model):
            os.remove(tmp_fn_model)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   tmp_fn_model)
            
        # validation
        if epoch%VALIDATE_INTERVAL == 0:
            t0 = time()
            sum_loss1 = 0.0; sum_loss2 = 0.0; sum_loss3 = 0.0;
            with torch.no_grad():
                model.eval()
                i = 0
                for _, rmsd, coord_r, coord_l, atom_r, atom_l, charge, n_model in validate_generator:
                    coord_l = coord_l[0].to(device=device,non_blocking=True)
                    coord_r = coord_r[0].to(device=device,non_blocking=True)
                    rmsd = rmsd[0].to(device=device,non_blocking=True)
                    atom_l = atom_l[0].to(device=device,non_blocking=True)
                    atom_r = atom_r[0].to(device=device,non_blocking=True)
                    charge = charge[0].to(device=device,non_blocking=True)
                    #
                    feature1, feature2 = generate_feature(atom_r, coord_r, coord_l)
                    # feed forward
                    energy_per_residue = model(feature1, feature2, atom_l, charge)
                    # model-wise sum
                    energy = energy_per_residue.sum(dim=1).view(-1)
                    loss1 = distinguish_global(energy, rmsd)
                    sum_loss1 += loss1.item()
                    loss2 = top1_success(energy, rmsd)
                    sum_loss2 += loss2#.item()
                    loss3 = top5_success(energy, rmsd)
                    sum_loss3 += loss3#.item()
                    i += 1
            t1 = time()
            print 'val, epoch: %3d loss: %.3f %.3f %.3f time: %.2f min'%(epoch, sum_loss1/i, sum_loss2/i, sum_loss3/i, (t1-t0)/60.)
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       'model/%s_%03d.pt'%(run_name, epoch))

def valid(fn_model_s):
    model_s = list()
    for fn_model in fn_model_s:
        model = MyModel().to(device)
        checkpoint = torch.load(fn_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_s.append(model)
    train_s, val_s = set_prep()
    #val_s.extend(train_s)
    validate_set = MyDataset(val_s,False)
    validate_generator = DataLoader(validate_set, **VAL_LOADER_PARAMS)
    
    with torch.no_grad():
        model.eval()
        result = dict()
        for tar, rmsd, coord_r, coord_l, atom_r, atom_l, charge, n_model in validate_generator:
            n_model = n_model.item()
            tar = tar[0]
            coord_l = coord_l[0].to(device=device,non_blocking=True)
            coord_r = coord_r[0].to(device=device,non_blocking=True)
            rmsd = rmsd[0].to(device=device,non_blocking=True)
            atom_l = atom_l[0].to(device=device,non_blocking=True)
            atom_r = atom_r[0].to(device=device,non_blocking=True)
            charge = charge[0].to(device=device,non_blocking=True)
            #
            feature1, feature2 = generate_feature(atom_r, coord_r, coord_l)
            # feed forward
            energy_s = list()
            for model in model_s:
                energy_per_residue = model(feature1, feature2, atom_l, charge)
                # model-wise sum
                energy = energy_per_residue.sum(dim=1).view(-1)
                energy_s.append(energy)
            # model-ensemble average
            energy = torch.stack(energy_s).mean(dim=0)
            idx = torch.topk(energy, 5, largest=False)[1]
            print '%s | %.3f %.3f'%(tar, rmsd[idx[0]].item(), rmsd[idx].min().item())
    return

def print_weight(fn_model):
    model = MyModel().to(device)
    checkpoint = torch.load(fn_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    fout = open('param.data','w')
    model.eval()
    with torch.no_grad():
        amino = torch.arange(21)
        tmp = model.fc1(amino)
    #
    for i in range(len(STANDARD)):
        fout.write('@ RESIDUE %-3s 8\n'%(STANDARD[i]))
        for j in range(8):
            fout.write('%10.7f\n'%(tmp[i,j].item()))
    #
    w = model.fc2_0.weight
    fout.write('@ LAYER WEIGHT FC2_0 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc2_0.bias
    fout.write('@ LAYER BIAS FC2_0 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc2_1.weight
    fout.write('@ LAYER WEIGHT FC2_1 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc2_1.bias
    fout.write('@ LAYER BIAS FC2_1 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc2_2.weight
    fout.write('@ LAYER WEIGHT FC2_2 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc2_2.bias
    fout.write('@ LAYER BIAS FC2_2 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc2_3.weight
    fout.write('@ LAYER WEIGHT FC2_3 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc2_3.bias
    fout.write('@ LAYER BIAS FC2_3 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc2_4.weight
    fout.write('@ LAYER WEIGHT FC2_4 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc2_4.bias
    fout.write('@ LAYER BIAS FC2_4 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc3_0.weight
    fout.write('@ LAYER WEIGHT FC3_0 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc3_0.bias
    fout.write('@ LAYER BIAS FC3_0 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc3_1.weight
    fout.write('@ LAYER WEIGHT FC3_1 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc3_1.bias
    fout.write('@ LAYER BIAS FC3_1 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc3_2.weight
    fout.write('@ LAYER WEIGHT FC3_2 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc3_2.bias
    fout.write('@ LAYER BIAS FC3_2 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc3_3.weight
    fout.write('@ LAYER WEIGHT FC3_3 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc3_3.bias
    fout.write('@ LAYER BIAS FC3_3 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc3_4.weight
    fout.write('@ LAYER WEIGHT FC3_4 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    b = model.fc3_4.bias
    fout.write('@ LAYER BIAS FC3_4 %3d\n'%(b.size()))
    for i in range(b.size()[0]):
        fout.write('%10.7f\n'%(b[i].item()))
    #
    w = model.fc4.weight
    fout.write('@ LAYER WEIGHT FC4_0 %3d %3d\n'%(w.size()))
    for i in range(w.size()[0]):
        for j in range(w.size()[1]):
            fout.write('%10.7f\n'%(w[i,j].item()))        
    #b = model.fc4.bias
    #for i in range(b.size()[0]):
    #    fout.write('%10.7f\n'%(b[i].item()))
    #
    fout.close()

    return

main()            
