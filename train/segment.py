import os
import numpy as np
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from utils import *
from dataset.sequence import SegTrajectoryDataset

logger = logging.getLogger(__name__)

def kl_divergence(p_mean, p_var, q_mean, q_var):
    return np.log(q_var / p_var) + (p_var**2 + (p_mean - q_mean)**2) / (2 * q_var**2) - 0.5

@hydra.main(config_path='../config', config_name='segment')
def main(cfg: DictConfig):
    logger.info("Loading models from checkpoint...")
    home_dir = os.path.join(os.path.expanduser("~"), "demo")
    model_value, _ = load_model(os.path.join(home_dir, cfg.ckpt_value), epoch='latest', device=f"cuda:{cfg.device}")
    model_return, _ = load_model(os.path.join(home_dir, cfg.ckpt_return), epoch='latest', device=f"cuda:{cfg.device}")
    set_parameter_requires_grad(model_value, requires_grad=False)
    set_parameter_requires_grad(model_return, requires_grad=False)
    model_value.eval()
    model_return.eval()  
    
    logger.info("Loading data...")
    with open(os.path.join(os.path.join(home_dir, cfg.data_path)), 'rb') as f:
        data = dict(np.load(f))
    dataset = SegTrajectoryDataset(
        sequence_length=cfg.subsampled_sequence_length,
        step=cfg.step,
        discount=cfg.discount,
        max_path_length=cfg.max_path_length,
        dataset=data,
        timeouts=cfg.timeouts,
    )
    
    logger.info("Evaluating KL uncertainty...")
    kls = np.zeros([len(dataset),cfg.max_path_length])
    for it, traj in enumerate(dataset):
        logger.info(f"Trajectory {it+1}/{len(dataset)}")
        pred_index = -1
        for index in range(traj['traj_length']-1):
            logger.info(f"Step {index+1}/{traj['traj_length']-1}")
            value_inputs = {
                'observations': to_torch(traj['observations'][max(index-cfg.max_window+1, 0):index+1].unsqueeze(0), device=f"cuda:{cfg.device}"),
                'actions': to_torch(traj['actions'][max(index-cfg.max_window+1, 0):index+1].unsqueeze(0), device=f"cuda:{cfg.device}")
            }
            return_inputs = {
                'observations': to_torch(traj['observations'][max(index-cfg.max_window+1, 0):index+2].unsqueeze(0), device=f"cuda:{cfg.device}"),
                'actions': to_torch(traj['actions'][max(index-cfg.max_window+1, 0):index+2].unsqueeze(0), device=f"cuda:{cfg.device}")
            }
            pred_V = torch.mean(torch.cat([output['mean'][0][-1].cpu() for output in model_value(value_inputs)]))
            pred_R = torch.mean(torch.cat([output['mean'][0][-1].cpu() for output in model_return(return_inputs)]))
            value_uncertainties = model_value.evaluate_uncertainty(value_inputs)[0][-1].cpu()
            return_uncertainties = model_return.evaluate_uncertainty(return_inputs)[0][-1].cpu()
            kl = kl_divergence(pred_V, value_uncertainties, pred_R, return_uncertainties)
            kls[it,index] = kl.item()
    
    logger.info("Segmenting trajectories...")
    kldis = np.zeros((kls.shape[0], cfg.max_path_length))
    path_length = dataset.path_lengths
    for it in range(kldis.shape[0]):
        old_index = -cfg.min_length
        for index in range(path_length[it]):
            if kls[it, index] >= cfg.threshold:
                if index - old_index >= cfg.min_length:
                    kldis[it, max(old_index, 0):max(old_index+cfg.min_length, 0)] = 0
                    kldis[it, max(old_index+cfg.min_length, 0):index] = np.arange(1, index-max(old_index+cfg.min_length, 0)+1)[::-1]
                else:
                    kldis[it, max(old_index, 0):index] = 0
                old_index = index
        if index - old_index >= cfg.min_length:
            kldis[it, max(old_index, 0):max(old_index+cfg.min_length, 0)] = 0
            kldis[it, max(old_index+cfg.min_length, 0):index+1] = np.arange(1, index-max(old_index+cfg.min_length, 0)+2)[::-1]
        else:
            kldis[it, max(old_index, 0):index+1] = 0
        kldis[it, index+1:] = 0
    np.save(os.path.join(home_dir, cfg.save_path, "index.npy"), kldis)

if __name__ == '__main__':
    main()
    logger.info('segment.py DONE!')