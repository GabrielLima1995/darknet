import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import get


cfg = 'cfg/'
confusion_matrix,fp,tp = get().matrix()
names = pd.read_csv(cfg+"buracos.names",header=None).values
names = np.append(names,['None'])
class_size =len(names)
array = np.r_[0:len(names)]

fig, ax = plt.subplots()
min_val, max_val = class_size,class_size
ax.matshow(confusion_matrix, cmap=plt.cm.Oranges)
x_pos =np.arange(class_size)
plt.xticks(x_pos,array)
plt.yticks(x_pos,array)

for i in range(class_size):
    for j in range(class_size):
        c = confusion_matrix[j,i]
        ax.text(i, j, str(int(c)), va='center', ha='center')

plt.savefig(cfg+'matrix_confusion.jpg')

#print("tp",tp)
#print("fp",fp)
#print("matrix",confusion_matrix)