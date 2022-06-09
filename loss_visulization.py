# This script is used to visualize loss curve in tensorboard
import pickle
from io_utils import model_dict, parse_args
import configs
import os
from torch.utils.tensorboard import SummaryWriter

params = parse_args('train')
params.method = 'matchingnet'
params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
if params.train_aug:
    params.checkpoint_dir += '_aug'
if not params.method  in ['baseline', 'baseline++']: 
    params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

save_path = os.path.join(params.checkpoint_dir, 'loss_list.pkl')
with open(save_path, 'rb') as f:
    loss_list = pickle.load(f)
writer = SummaryWriter()
for i in range(len(loss_list)):
    writer.add_scalar(save_path, loss_list[i],i)
writer.close()

