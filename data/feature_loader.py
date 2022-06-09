import torch
import numpy as np
import h5py

class SimpleHDF5Dataset:
    ## extract the stored feature
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0 
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
           # print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def init_loader(filename):
    """
    To get the saved feature after training. used in test.py.
    Input:
        filename : the file name of saved feature
    Output:
        cl_data_file : the saved feature used in testing process
    """
    with h5py.File(filename, 'r') as f:
        fileset = SimpleHDF5Dataset(f)

    #labels = [ l for l  in fileset.all_labels if l != 0]
    feats = fileset.all_feats_dset
    labels = fileset.all_labels
    while np.sum(feats[-1]) == 0:
        feats  = np.delete(feats,-1,axis = 0)
        labels = np.delete(labels,-1,axis = 0)
        
    class_list = np.unique(np.array(labels)).tolist() 

    cl_data_file = {}
    for cl in class_list:
        cl_data_file[cl] = []
    for ind in range(len(labels)):
        cl_data_file[labels[ind,0]].append( feats[ind])

    return cl_data_file
