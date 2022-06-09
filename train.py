# This script is used to train model with few-shot methods
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
import pickle
import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from torch.autograd import Variable
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.FRN import FRN
from io_utils import model_dict, parse_args, get_resume_file  

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=params.weightdecay)
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001,weight_decay=params.weightdecay,momentum=0.9)
    else:
       raise ValueError('Unknown optimization, please define by yourself')
    
    max_acc = 0       
    store_acc = []
    loss_list = []
    save_path = os.path.join(params.checkpoint_dir, ''.join(['loss_list','.pkl']))
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        loss = model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        loss_list.append(loss)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
        store_acc.append(acc)
        if acc > max_acc : #for baseline and baseline++, we don't use validation in default and we let acc = -1, but we allow options to validate with DB index
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    with open(save_path, 'wb') as f:
        pickle.dump(loss_list, f)
    
    ## These code is used to store the validation accuracy during training
    # save_path = os.path.join(params.checkpoint_dir, ''.join(['store_acc_',params.opti,str(params.weightdecay),'.pkl']))
    # with open(save_path, 'wb') as f:
    #     pickle.dump(store_acc, f)
    return model

if __name__=='__main__':
    torch.multiprocessing.set_start_method('spawn')
    np.random.seed(10)
    params = parse_args('train')
    params.dataset = 'modelnet40'

    if params.dataset == 'modelnet40':
        base_file = configs.data_dir['modelnet40'] + 'base.json' 
        val_file = configs.data_dir['modelnet40'] + 'novel.json'
        params.num_classes = 30

    optimization = 'Adam'

    if params.stop_epoch == -1: 
        params.stop_epoch = 101
    if params.method == 'relationnet_softmax':
        params.stop_epoch = 301

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(batch_size = 16)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(batch_size = 16)
        val_loader      = val_datamgr.get_data_loader( base_file, aug = False)

        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'FRN']:
        n_query = max(1, int(15* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small

        train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False)  

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            feature_model = model_dict[params.model]
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'
            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method == 'FRN':
            model           = FRN( model_dict[params.model], **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'
    if not params.method  in ['baseline', 'baseline++']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch 

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    elif params.warmup: #We also support warmup from pretrained baseline feature, but we never used in our paper
        baseline_checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, 'baseline')
        if params.train_aug:
            baseline_checkpoint_dir += '_aug'
        warmup_resume_file = get_resume_file(baseline_checkpoint_dir)
        tmp = torch.load(warmup_resume_file)
        if tmp is not None: 
            state = tmp['state']
            state_keys = list(state.keys())
            for i, key in enumerate(state_keys):
                if "feature." in key:
                    newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                    state[newkey] = state.pop(key)
                else:
                    state.pop(key)
            model.feature.load_state_dict(state)
        else:
            raise ValueError('No warm_up file')

    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)