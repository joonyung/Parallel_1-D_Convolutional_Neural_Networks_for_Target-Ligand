import os

list_file = open("/home/jy2/practice/pytorch/dud-e/list").read().splitlines()
wr_file = open("protein_seq","w")


for ln in list_file:
    cmd = "pdb_seq /home/jy2/practice/pytorch/dud-e/%s/receptor.pdb >> protein_seq"%ln
    os.system(cmd)


