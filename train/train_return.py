import os
import numpy as np
import torch
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import utils
from utils import *
from dataset.trajectory import EnsembleTrajectoryDataset, EnsembleHorizonTrajectoryDataset
from model.unrest import EnsembleReturnGPT, ReturnGPT

logger = logging.getLogger(__name__)

@hydra.main(config_path='../config', config_name='train_return')
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
    
    dataset_config = utils.Config(
        EnsembleHorizonTrajectoryDataset if cfg.horizon else EnsembleTrajectoryDataset,
        ensemble_size = cfg.ensemble_size, 
        mask_prob = cfg.mask_prob,
        savepath=('data_config.pkl'),
        env=None,
        dataset=data,
        penalty=cfg.termination_penalty,
        sequence_length=sequence_length,
        step=cfg.step,
        discount=cfg.discount,
        horizon=cfg.horizon,
        max_path_length=5000,
        timeouts=False,
        anystep=cfg.anystep,
    )
    
    dataset = dataset_config()
    obs_dim = dataset.observation_dim
    act_dim = dataset.action_dim
    transition_dim = 2
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
            EnsembleReturnGPT,
            observation_mean=to_torch(stats['observation_mean']),
            observation_std=to_torch(stats['observation_std']),
            action_mean=to_torch(stats['action_mean']),
            action_std=to_torch(stats['action_std']),
            savepath=('model_config.pkl'),
            # architecture
            block_size=block_size,
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd*cfg.n_head,
            # dimensions
            observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
            # dropout probabilities
            embd_pdrop=cfg.embd_pdrop, resid_pdrop=cfg.resid_pdrop, attn_pdrop=cfg.attn_pdrop,
            # ensemble size
            ensemble_size=cfg.ensemble_size,
            use_value=cfg.use_value, use_new_s=cfg.use_new_s, horizon=cfg.horizon, anystep=cfg.anystep
            )
        model = model_config()
        model.to(device)
    
    # trainer
    warmup_tokens = len(dataset) * block_size * cfg.mask_prob ## number of tokens seen per epoch
    final_tokens = 20 * warmup_tokens
    
    trainer_config = utils.Config(
        utils.EnsembleTrainer,
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
        ensemble_size=cfg.ensemble_size,
    )
    trainer = trainer_config()
    trainer.n_epochs = cur_epoch + 1

    # main loop
    ## scale number of epochs to keep number of updates constant
    n_epochs = int((4e6 / len(dataset) * cfg.n_epochs_ref))
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
    logger.info('train_return.py DONE!')