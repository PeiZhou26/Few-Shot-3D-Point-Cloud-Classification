# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'../'))
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod
from torch.utils.data import Sampler
from copy import deepcopy

def shuffle_points(pc):
    """
    Shuffle the points in one sample for data augmentation
    Input:
        pc : original point cloud
    Output:
        pc : point cloud
    """
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    return pc[idx,:]
def rotate_perturbation_point_cloud(pc, angle_sigma=0.06, angle_clip=0.18):
    """
    Rotating the point clouds for data augmentation
    Input:
        pc : original point cloud
        angle_sigma : the variance of the random angle 
        angle_clip: upper and lower limits of rotation angle
    Output:
        pc : point cloud
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                   [0,np.cos(angles[0]),-np.sin(angles[0])],
                   [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                   [0,1,0],
                   [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                   [np.sin(angles[2]),np.cos(angles[2]),0],
                   [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx)).astype(np.float32)
    pc = np.dot(pc, R)
    return pc
def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    """
    Randomly scale the point clouds for data augmentation
    Input:
        pc : original point cloud
        scale_low: lower limit of scale
        scale_high: upper limit of scale
    Output:
        pc : point cloud
    """
    scale = np.random.uniform(scale_low, scale_high, 1).astype(np.float32)
    pc *= scale
    return pc
def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """
    Randomly jitter the point clouds for data augmentation
    Input:
        pc : original point cloud
        sigma : the variance of jittering
        clip: upper and lower limits of jittering
    Output:
        pc : point cloud
    """
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(np.float32)
    jittered_data += pc
    return jittered_data


class point_trans(object):
    ## A combination of multiple data augmentation methods
    def __init__(self, point_shuffle = True, rotate = True, scale=True,jitter=True):
        self.point_shuffle = point_shuffle
        self.rotate = rotate
        self.scale = scale
        self.jitter = jitter

    def __call__(self, points):
        if self.point_shuffle:
            points = shuffle_points(points)
        if self.rotate:
            points = rotate_perturbation_point_cloud(points)
        if self.scale:
            points = random_scale_point_cloud(points)
        if self.jitter:
            points = jitter_point_cloud(points)
        return points


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    ## dataloader of standard tranfer learning
    def __init__(self, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        if aug:
            transform = point_trans()
            dataset = SimpleDataset(data_file, transform=transform)
        else:
            dataset = SimpleDataset(data_file)
        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 4, pin_memory = True,drop_last=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

class SetDataManager(DataManager):
    ## dataloader of standard meta learning, n-way k-shot for each episode
    def __init__(self, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        if aug:
            transform = point_trans()
            dataset = SetDataset( data_file , self.batch_size, transform )
        else:
            dataset = SetDataset( data_file , self.batch_size)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 8, pin_memory = True)  
        # num_worker can not be large     
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__=='__main__':
    # data_loader_params = dict(batch_sampler = 1,  num_workers = 12, pin_memory = True)
    # print(data_loader_params)
    # pass
    # print(PYTHONPATH)
    import multiprocessing as mp
    mp.set_start_method('spawn')
    from data.dataset import SimpleDataset
    pwd = os.path.abspath(os.path.dirname(__file__))
    base_name = os.path.abspath(os.path.join(pwd,'../..','dataset/modelnet40_normal_resampled/base.json')) 
    datasets = SetDataManager(5, 5, 15).get_data_loader(base_name, True)
    xx, yy = next(iter(datasets))   
    print(xx,yy)
   