import os
import numpy as np
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import utils
from utils import *
from dataset.trajectory import FixedHorizonTrajectoryDataset, IndicedHorizonDataset
from model.unrest import UNREST

logger = logging.getLogger(__name__)

@hydra.main(config_path='../config', config_name='train_unrest')
def main(cfg: DictConfig):
    # set up
    set_seed(cfg.seed)
    device = torch.device(f"cuda:{cfg.device}")
    logger.setLevel(getattr(logging, cfg.log_level.upper()))
    
    # dataset
    home_dir = os.path.join(os.path.expanduser("~"), "demo")
    data_file = os.path.join(home_dir, cfg.dataset_path)
    with open(data_file, 'rb') as f:
        data = dict(np.load(f))
        
    sequence_length = cfg.subsampled_sequence_length * cfg.step
    
    if cfg.indice_path:
        dataset_config = utils.Config(
            IndicedHorizonDataset,
            savepath=('data_config.pkl'),
            env=None,
            dataset=data,
            penalty=cfg.termination_penalty,
            sequence_length=sequence_length,
            step=cfg.step,
            discount=cfg.discount,
            horizon=cfg.horizon,
            anystep=cfg.anystep,
            max_path_length=cfg.max_path_length,
            certain_only=cfg.certain_only,
            uncertain_only=cfg.uncertain_only,
            timeouts=False,
            indice_path=os.path.join(home_dir, cfg.indice_path)
        )
    else:
        dataset_config = utils.Config(
            FixedHorizonTrajectoryDataset,
            savepath=('data_config.pkl'),
            env=None,
            dataset=data,
            penalty=cfg.termination_penalty,
            sequence_length=sequence_length,
            step=cfg.step,
            discount=cfg.discount,
            horizon=cfg.horizon,
            anystep=cfg.anystep,
            max_path_length=cfg.max_path_length,
            timeouts=False,
        )
    
    dataset = dataset_config()
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = 3 
    stats = dataset.get_stats()
    
    # model
    block_size = cfg.subsampled_sequence_length * transition_dim - 1
    logger.info(
        f'Dataset size: {len(dataset)} | '
        f'Joined dim: {transition_dim} '
        f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
    )
    
    if cfg.resume and cfg.ckpt:
        logger.info(f"Resume training from {cfg.ckpt}...")
        model, cur_epoch = utils.load_model(os.path.join(home_dir, cfg.ckpt), epoch=cfg.epoch, device=f"cuda:{cfg.device}")
    else:
        cur_epoch = -1
        model_config = utils.Config(
            UNREST,
            action_tanh=cfg.action_tanh,
            observation_mean=to_torch(stats['observation_mean']),
            observation_std=to_torch(stats['observation_std']),
            action_mean=to_torch(stats['action_mean']),
            action_std=to_torch(stats['action_std']),
            horizon_return_mean=to_torch(stats['horizon_return_mean']),
            horizon_return_std=to_torch(stats['horizon_return_std']),
            savepath=('model_config.pkl'),
            # architecture
            block_size=block_size,
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd*cfg.n_head,
            # dimensions
            observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
            # dropout probabilities
            embd_pdrop=cfg.embd_pdrop, resid_pdrop=cfg.resid_pdrop, attn_pdrop=cfg.attn_pdrop,
            # for fixed model
            horizon=dataset.horizon, max_ep_length=cfg.max_path_length+sequence_length,
            global_conditioned=cfg.global_conditioned, n_global_condition_return_token=cfg.n_global_condition_return_token,
            global_return_min=to_torch(stats['return_min']), global_return_max=to_torch(stats['return_max']), embd_time=cfg.embd_time,
            # for auto model, additionally
            global_predicted=cfg.global_predicted, n_global_predict_return_token=cfg.n_global_predict_return_token,
            horizon_return_min=to_torch(stats['horizon_return_min']), horizon_return_max=to_torch(stats['horizon_return_max']),
            n_horizon_predict_return_token=cfg.n_horizon_predict_return_token, horizon_return_weight=cfg.horizon_return_weight,
            global_return_weight=cfg.global_return_weight, action_weight=cfg.action_weight, device=device
        )
        model = model_config()
        model.to(device)
    
    # trainer
    warmup_tokens = len(dataset) * block_size ## number of tokens seen per epoch
    final_tokens = 20 * warmup_tokens
    
    trainer_config = utils.Config(
        utils.Trainer,
        savepath=('trainer_config.pkl'),
        # optimization parameters
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        betas=(0.9, 0.95),
        grad_norm_clip=1.0,
        weight_decay=0.1, # only applied on matmul weights
        # learning rate decay: linear warmup followed by cosine decay to 10% of original
        lr_decay=cfg.lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        ## dataloader
        num_workers=4,
        device=device,
        logger=logger,
    )
    trainer = trainer_config()
    trainer.n_epochs = cur_epoch + 1

    # main loop
    ## scale number of epochs to keep number of updates constant
    n_epochs = int((8e6 / len(dataset) * cfg.n_epochs_ref))
    for epoch in range(cur_epoch+1, n_epochs):
        logger.info(f'\nEpoch: {epoch} / {n_epochs} | {cfg.exp_name}')

        trainer.train(model, dataset, log_freq=cfg.log_freq)
        
        if epoch % cfg.save_freq == 0:
            ## get greatest multiple of `save_freq` less than or equal to `save_epoch`
            statepath = os.path.join(f'state_{epoch}.pt')
            logger.info(f'Saving model to {statepath}')

            ## save state to disk
            state = model.state_dict()
            torch.save(state, statepath)

if __name__ == '__main__':
    main()
    logger.info('train_unrest.py DONE!')