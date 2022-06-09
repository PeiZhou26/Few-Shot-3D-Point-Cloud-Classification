# This code is used to preprocess shapenet, but shapenet is not used in this project
import os
from tqdm import tqdm
import csv
import numpy as np

shapenetpath = os.path.abspath(os.getcwd()+'/../dataset/shapenetsem') 

def del_file(filepath):
    files = os.listdir(filepath)
    for file in tqdm(files):
        if '.' in file:
            suffix = file.split('.')[1]
            prefix = file.split('.')[0]
            if suffix != 'obj' or prefix[0:4]=='room':
                os.remove(os.path.join(filepath, file))

def conver_obj2txt(filepath):
    # first generate a dict to store the class for each sample
    metadata_path = os.path.join(shapenetpath,'metadata.csv')
    store_dict = {}
    lines = csv.reader(open(metadata_path))
    for line in lines:
        para = line 
        if para[0] == 'fullId' or para[1] =='':
            continue
        file_num = para[0].split('.')[1]
        class_num = para[1]  
        store_dict[file_num] = class_num
    keys = np.unique(np.array((list(store_dict.values()))))
    xx = 1
    class_list = []
    sample_list = []
    files = os.listdir(filepath)
    for file in tqdm(files):
        point_set = []
        file_name = os.path.join(filepath, file)
        with open(file_name,'r') as f:
            lines = f.readlines()
        for line in lines:
            para = line.split()
            if para[1] == 'Model':
                if not para[2] in store_dict.keys():
                    break
                category = store_dict[para[2]]
                if not category in class_dict.keys():
                    class_dict[category] = []
                new_filename = '_'.join([category,str(len(class_dict[category]))])+'.dat'
            class_list.append(store_dict[para[2]])
            sample_list.append(new_filename)    
  
            if para[0] == 'v':
                point = []
                point.append(float(para[1]))
                point.append(float(para[2]))
                point.append(float(para[3]))
                point_set.append(point)

        with open(new_filename, 'w') as out:
            for j in range(len(point_set)):
                out.write(str(vset[j][0]) + " " + str(vset[j][1]) + " " + str(vset[j][2]) + "\n")


