# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
from tqdm import tqdm
import pickle
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:int(x)

def pc_normalize(pc):
    """
    Normalize the point cloud data
    Input:
        pc : non-normalized point cloud
    Output:
        pc : normalized point cloud
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point[:,0:3]

class SimpleDataset:
    ## dataset for standard transfer learning
    def __init__(self, data_file, npoints=2048,preprocess=False,transform=None):
        assert isinstance(npoints, int)
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.datafile = data_file
        self.npoints = npoints
        self.preprocess = preprocess
        self.transform = transform
        _,save_path = os.path.split(data_file)
        save_path = save_path.split('.')[0]
        self.save_path = save_path = os.path.abspath(self.datafile+'/../'+'model40'+save_path+'.dat')
        if self.preprocess:
            self.list_of_points = [None] * len(self.meta['image_names'])
            self.list_of_labels = [None] * len(self.meta['image_names'])
            for i in tqdm(range(len(self.meta['image_names']))):
                image_path = os.path.abspath(self.datafile+'/../'+self.meta['image_names'][i])
                point_set = np.loadtxt(image_path, delimiter=',').astype(np.float32)
                point_set = pc_normalize(farthest_point_sample(point_set, self.npoints)) 
                self.list_of_points[i] = point_set
                self.list_of_labels[i] = int(self.meta['image_labels'][i])
            with open(save_path, 'wb') as f:
                pickle.dump([self.list_of_points, self.list_of_labels], f)
        else:
            with open(save_path, 'rb') as f:
                self.list_of_points, self.list_of_labels = pickle.load(f)
                
    def __getitem__(self, i):
        if os.path.exists(self.save_path):
            point_set, target = self.list_of_points[i], self.list_of_labels[i]
        else:
            image_path = os.path.abspath(self.datafile+'/../'+self.meta['image_names'][i])
            point_set = np.loadtxt(image_path, delimiter=',').astype(np.float32)
            point_set = point_set[0:self.npoints, 0:3]
            target = self.meta['image_labels'][i]
        if self.transform is not None: 
            point_set = self.transform(point_set) 
        point_set = torch.from_numpy(point_set)
        target = torch.from_numpy(np.array([target]))
        return point_set, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    ## dataset for meta learning
    def __init__(self, data_file, batch_size, transform=None):
        _,save_path = os.path.split(data_file)
        save_path = save_path.split('.')[0]
        save_path = os.path.abspath(data_file+'/../'+'model40'+save_path+'.dat')
        with open(save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)
    
        self.cl_list = np.unique(self.list_of_labels).tolist()
        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []
        
        for x,y in zip(self.list_of_points, self.list_of_labels):
            self.sub_meta[y].append(x)

        self.sub_dataloader = []
        self.subdataset = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  drop_last = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.subdataset.append(sub_dataset)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    ## subdataset for each class in meta-learning
    def __init__(self, sub_meta, cl, transform):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform

    def __getitem__(self,i):
        point_set = self.sub_meta[i]
        if self.transform is not None:  
            point_set = self.transform(point_set)
        target = self.cl
        point_set = torch.from_numpy(point_set)
        target = torch.from_numpy(np.array([target]))
        return point_set, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    ## sampler used in dataloader for episode learning
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]



if __name__=='__main__':
    import json
    import datamgr
    from copy import deepcopy

    pwd = os.path.abspath(os.path.dirname(__file__))
    print('start')
    base_name = os.path.abspath(os.path.join(pwd,'../..','dataset/modelnet40_normal_resampled/base.json')) 
    dataset = SimpleDataset(base_name,preprocess=True)
    base_name = os.path.abspath(os.path.join(pwd,'../..','dataset/modelnet40_normal_resampled/novel.json')) 
    dataset = SimpleDataset(base_name,preprocess=True)
    

