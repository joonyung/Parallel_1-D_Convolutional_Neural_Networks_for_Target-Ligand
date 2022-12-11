import os

file_list = open("/home/jy2/practice/pytorch/data_split/val_list_2").read().splitlines()
wr_file = open("seq_val_list_2","w")

for ln in file_list:
    cmd = "pdb_seq /home/jy2/practice/pytorch/dud-e/%s/receptor.pdb >> seq_val_list_2"%ln
    os.system(cmd)


