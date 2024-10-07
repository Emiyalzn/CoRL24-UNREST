import os
import importlib
import random
import numpy as np
import torch
import glob
import pickle

def to_torch(x, dtype=None, device=None):
    dtype = dtype or torch.float
    device = device or 'cuda:0'
    return torch.tensor(x, dtype=dtype, device=device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def set_parameter_requires_grad(model, requires_grad):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = requires_grad
            
def get_latest_epoch(loadpath):
    states = glob.glob1(loadpath, 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch
    
def load_model(*loadpath, epoch=None, device='cuda:0', eval=False):
    loadpath = os.path.join(*loadpath)
    config_path = os.path.join(loadpath, 'model_config.pkl')

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'[ utils/serialization ] Loading model epoch: {epoch}')
    state_path = os.path.join(loadpath, f'state_{epoch}.pt')

    config = pickle.load(open(config_path, 'rb'))
    config['eval'] = eval
    config['device'] = device

    state = torch.load(state_path, map_location=device)
    if eval:
        filtered_state = {k: v for k, v in state.items() if not k.startswith('return_gpt')}
    else:
        filtered_state = state

    model = config()
    model.load_state_dict(filtered_state, strict=True)
    model.to(device)

    print(f'\n[ utils/serialization ] Loaded config from {config_path}\n')
    print(config)

    return model, epoch