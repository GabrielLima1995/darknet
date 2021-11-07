import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import get
import argparse

parser = argparse.ArgumentParser(description='Confusion_matrix generator')

parser.add_argument("-m", "--main", required=True,
help="main path is the directory which has the image list and name file ")

parser.add_argument("-n", "--names_file", required=True,
help="name of file names ")

parser.add_argument("-i", "--img_list_name", required=True,
	help="image's list name (This list must have only images path) ")

parser.add_argument("-f", "--img_folder_path", required=True,
	help="image folder path without last slash")

args = vars(parser.parse_args())

confusion_matrix,fp,tp = get().matrix(args['img_folder_path'],args['main'],args['img_list_name'])
names = pd.read_csv("{}/{}".format(args['main'],args['names_file']),header=None).values
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

plt.savefig('{}/confusion_matrix.jpg'.format(args['main']))

#print("tp",tp)
#print("fp",fp)
#print("matrix",confusion_matrix)