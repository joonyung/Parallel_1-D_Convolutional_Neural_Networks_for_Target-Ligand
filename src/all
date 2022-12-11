import os
import random



target_list = open('/home/jy2/practice/pytorch/dud-e/list').read().splitlines()
protein_seq = open('/home/jy2/practice/pytorch/protein_seq').read().splitlines()

dict_amid = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, "K": 10, 
        "M": 11, "L": 12, "O": 13, "N": 14, "Q":15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20,
        "W": 21, "V": 22, "Y":23, "X": 24, "Z": 25}

#making SMILES encoding dict
charsmilist = open('/home/jy2/practice/pytorch/dud-e/total_char_smi').read().splitlines()

char_i = 1
dict_char_smi = {}
for char in charsmilist:
    dict_char_smi[char.split()[0]] = char_i
    char_i += 1

ligMAXLEN = 200
proMAXLEN = 580  #maximum 580

dict_protein_seq = {}

def get_dict_seq(file_protein_seq):
    name = ""
    for ln in file_protein_seq:
        if ln.startswith('>'):
            name = ln.replace("> /home/jy2/practice/pytorch/dud-e/","")
            name = name.replace("/receptor.pdb:chain","")
            name = name.replace("A","")
            name = name.replace("B","")
            name = name.replace(" ","", 2)
        else:
            if name not in dict_protein_seq.keys():
                dict_protein_seq[name] = ln
            else:
                dict_protein_seq[name] = dict_protein_seq[name] + ln

get_dict_seq(protein_seq)

def seq_enc(tar_seq):
    seq_list = []
    for char in tar_seq:
        seq_list.append(dict_amid[char])
    if len(seq_list) < proMAXLEN:
        for i_iter in range(proMAXLEN - len(seq_list)):
            seq_list.append(0)
    elif len(seq_list) > proMAXLEN:
        del seq_list[proMAXLEN-1:]

    return seq_list

def smi_enc(lig_smi):
    encsmi_list = []
    for smi in lig_smi:
        tmp_list = []
        for char in smi.split()[0]:
            tmp_list.append(dict_char_smi[char])
        if len(tmp_list) < ligMAXLEN:
            for i_iter in range(ligMAXLEN - len(tmp_list)):
                tmp_list.append(0)
        elif len(tmp_list) > ligMAXLEN:
            del tmp_list[ligMAXLEN-1:]
        encsmi_list.append(tmp_list)
    
    return encsmi_list

def combine_data(target, actives, inactives):
    datapair = []
    for act_lig in actives:
        datapair.append([target, act_lig, 1])
    for inact_lig in inactives:
        datapair.append([target, inact_lig, 0])
    
    return datapair

def get_data_set():
    total_data = []
    
    for target in dict_protein_seq.keys():
        actives_smi = open('/home/jy2/practice/pytorch/dud-e/%s/actives_final.ism'%target).read().splitlines()
        inactives_smi = open('/home/jy2/practice/pytorch/dud-e/%s/inactives_nM_chembl.ism'%target).read().splitlines()
    
        enc_target = seq_enc(dict_protein_seq[target])
        enc_actives = smi_enc(actives_smi)
        enc_inactives = smi_enc(inactives_smi)

        target_data = combine_data(enc_target, enc_actives, enc_inactives)
        total_data = total_data + target_data
    
    return total_data

data_set = get_data_set()


'''
#random.shuffle(data_set)

train_set = []
test_set = []
div_ratio = 6

def divide_train_test(total_data):
    num_of_train = (len(total_data)/6)*5
    train_set[0:num_of_train] = total_data[0:num_of_train]
    test_set[num_of_train:] = total_data[num_of_train:]

#divide_train_test(data_set)
'''

'''
#custom dataset
import torch
from torch.utils.data import Dataset, DataLoader
class BCFdata(Dataset):
    def __init__(self, pro_input, lig_input, bind_label):
        super(BCFdata, self).__init__()

        self.pro = torch.tensor(pro_input, dtype = torch.float)
        self.pro = self.pro.unsqueeze(0).unsqueeze(0)
        self.lig = torch.tensor(lig_input, dtype = torch.float)
        self.lig = self.lig.unsqueeze(0).unsqueeze(0)
        self.label = torch.tensor([bind_label], dtype = torch.float)
        self.label = self.label.unsqueeze(0).unsqueeze(0)



    def __getitem__(self, index):
        return self.pro[index], self.lig[index], self.label[index]


    def __len__(self):
        return len(self.label)
'''
