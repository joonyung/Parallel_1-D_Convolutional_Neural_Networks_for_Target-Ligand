import os

target_list = open('/home/jy2/practice/pytorch/data_split/target_list').read().splitlines()
protein_seq = open('/home/jy2/practice/pytorch/protein_seq').read().splitlines()

train_list_0 = open('/home/jy2/practice/pytorch/data_split/train_list_0','w')
train_list_1 = open('/home/jy2/practice/pytorch/data_split/train_list_1','w')
train_list_2 = open('/home/jy2/practice/pytorch/data_split/train_list_2','w')
train_list_3 = open('/home/jy2/practice/pytorch/data_split/train_list_3','w')
train_list_4 = open('/home/jy2/practice/pytorch/data_split/train_list_4','w')
train_list_5 = open('/home/jy2/practice/pytorch/data_split/train_list_5','w')

val_list_0 = open('/home/jy2/practice/pytorch/data_split/val_list_0','w')
val_list_1 = open('/home/jy2/practice/pytorch/data_split/val_list_1','w')
val_list_2 = open('/home/jy2/practice/pytorch/data_split/val_list_2','w')
val_list_3 = open('/home/jy2/practice/pytorch/data_split/val_list_3','w')
val_list_4 = open('/home/jy2/practice/pytorch/data_split/val_list_4','w')
val_list_5 = open('/home/jy2/practice/pytorch/data_split/val_list_5','w')

list_train_list = [train_list_0, train_list_1, train_list_2, train_list_3, train_list_4, train_list_5]
list_val_list = [val_list_0, val_list_1, val_list_2, val_list_3, val_list_4, val_list_5]

FOLD_NUM = 6

list_index = 0
for ln in target_list:
    residue = list_index % FOLD_NUM
    for idx in range(6):
        if residue == idx:
            list_val_list[idx].write(ln)
            list_val_list[idx].write("\n")
            for element in list_train_list:
                if list_train_list.index(element) != idx:
                    element.write(ln)
                    element.write("\n")
    list_index += 1 

