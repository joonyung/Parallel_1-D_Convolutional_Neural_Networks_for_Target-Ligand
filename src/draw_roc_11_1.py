import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc, roc_curve

with open("roc_label_true_11_1.txt", "rb") as path_true:
    label_true = pickle.load(path_true)
with open("roc_label_score_11_1.txt", "rb") as path_score:
    label_score = pickle.load(path_score)

label_true = np.array(label_true)
label_score = np.array(label_score)

fpr, tpr, thresholds = roc_curve(label_true, label_score)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
print('0000')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' %roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('Flase Positive RAte')

plt.savefig('model11_1.png')
