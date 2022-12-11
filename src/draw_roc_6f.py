import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve

N_FOLD = 6
dict_true = {}
dict_score = {}


for idx in range(N_FOLD):
    with open("roc_label_true_6f_%i.txt"%idx, "rb") as path_true:
        label_true = pickle.load(path_true)
        label_true = np.array(label_true)
        dict_true[idx] = label_true
    with open("roc_label_score_6f_%i.txt"%idx, "rb") as path_score:
        label_score = pickle.load(path_score)
        label_score = np.array(label_score)
        dict_score[idx] = label_score

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure()
for idx in range(N_FOLD):
    fpr[idx], tpr[idx], _ = roc_curve(dict_true[idx], dict_score[idx])
    roc_auc[idx] = auc(fpr[idx], tpr[idx])
    plt.plot(fpr[idx], tpr[idx], label = 'FOLD %i (AUC = %0.4f)'%(idx, roc_auc[idx]))

plt.plot([0,1],[0,1], linestyle='--', lw = 2, color = 'r', label = 'Random guess (AUC = 0.5)', alpha = 0.3)
plt.title('6-fold Cross Validation ROC')
plt.legend(loc = 'lower right')
#plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('Flase Positive Rate')

plt.savefig('ROC_6f.png')
