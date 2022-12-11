import os

output_smi = file('/home/jy2/practice/pytorch/dud-e/total_char_smi','w')
target_list = file('/home/jy2/practice/pytorch/dud-e/list').read().splitlines()

dict_smi = {}
ls_actdec = ["actives_final.ism", "inactives_nM_chembl.ism","decoys_final.ism"]

for target in target_list:
    for lig_type in ls_actdec:
        input_smi = file('/home/jy2/practice/pytorch/dud-e/%s/%s'%(target, lig_type),'r').read().splitlines()

        for line in input_smi:
            for char in line.split()[0]:    
                if char in dict_smi:
                    dict_smi[char] += 1
                else:
                    dict_smi[char] = 1


for key, val in dict_smi.items():
    output_smi.write('%s %s\n'%(key, val))

#problem ampc
