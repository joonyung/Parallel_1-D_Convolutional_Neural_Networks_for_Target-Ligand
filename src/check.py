import os
import glob
import numpy as np

# adrb2 3ny8

#for i in glob.glob('pytorch/dud-e/*/receptor.pdb'):
#    os.system('pdb_seq %s >> tmp.fa'%i)
#f=file('tmp1.fa','w')
#for i in file('tmp.fa'):
#    if '> pytorch' in i:
#        f.write('> %s\n'%i.split('/')[-2])
#    else:
#        f.write(i)
#f.close()


dic={}
for i in range(6):
#    dic[i]=[]
    for ln in file('/home/jy2/practice/pytorch/data_split/val_list_%i'%i):
        dic[ln.strip()] = i
#        dic[i].append(ln.strip())
'''
#list=[]
#for i in file('cl70.clstr'):
#    if '>Cluster' in i:
#        if len(list) > 1:
            for j in list:
                print num, j, dic[j]
        list=[]
        num=i.split()[-1]
    else:
        list.append(i.split()[3].split('.')[0])
'''
list=[]
tmp=[]
tmp1=[]
a_act=0
a_inact=0
for i in file('/home/jy2/practice/pytorch/dud-e/list'):
    i=i.strip()
    inact = len(file('/home/jy2/practice/pytorch/dud-e/%s/inactives_nM_chembl.ism'%i).readlines())
    decoy = len(file('/home/jy2/practice/pytorch/dud-e/%s/decoys_final.ism'%i).readlines())
    act = len(file('/home/jy2/practice/pytorch/dud-e/%s/actives_final.ism'%i).readlines())
    inact+=decoy
    #
    rat = float(inact)/float(act)
    a_act+=act
    a_inact+=inact
    tmp.append(inact)
    tmp1.append(inact)
    list.append(rat)
print 'rate = inactive/active'
print 'rate_ave', np.average(list), np.median(list), np.max(list), np.min(list)
print 'active', np.average(tmp1), np.median(tmp1), np.max(tmp1), np.min(tmp1)
print 'inactive', np.average(tmp), np.median(tmp), np.max(tmp), np.min(tmp)

print 'tot_num_rate', float(a_inact)/float(a_act)
print '#'

for num in range(6):
    list=[0,0]
    for i in file('/home/jy2/practice/pytorch/data_split/val_list_%i'%num):
        i=i.strip()
        inact = len(file('/home/jy2/practice/pytorch/dud-e/%s/inactives_nM_chembl.ism'%i).readlines())
        act = len(file('/home/jy2/practice/pytorch/dud-e/%s/actives_final.ism'%i).readlines())
        decoy = len(file('/home/jy2/practice/pytorch/dud-e/%s/decoys_final.ism'%i).readlines())
        inact+=decoy
        #
        list[0]+=act
        list[1]+=inact
    print ('cross_val', num, list[0], list[1], float(list[1])/float(list[0]))

