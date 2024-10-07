import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from .gpt import GPT

class ReturnUniformDiscretizer:
    def __init__(self, lower, upper, n_bins, onehot=False):
        self.lb = lower
        self.ub = upper
        self.n_bins = n_bins
        self.onehot = onehot
        self.bin_width = (self.ub - self.lb) / self.n_bins
    
    def discretize(self, x):
        indices = torch.clip(torch.floor((x - self.lb) / self.bin_width), 0, self.n_bins-1).to(torch.int64)
        if self.onehot:
            indices = torch.zeros(list(x.shape[:-1])+[self.n_bins]).to(x.device).scatter(-1, indices, 1)
        return indices
    
    def reconstruct(self, indices):
        if self.onehot:
            indices = torch.argmax(indices, dim=-1, keepdims=True)
        x = self.lb + self.bin_width/2 + indices * self.bin_width
        return x.cpu().numpy()

class ReturnGPT(GPT):
    def __init__(self, config):
        if not hasattr(config, 'output_dim'):
            config.output_dim = config.n_embd
        config.mask_values = False
        self.anystep = config.anystep if hasattr(config, 'anystep') else False
        self.horizon = config.horizon if hasattr(config, 'horizon') else 100
        
        super().__init__(config)
        
        self.observation_mean = nn.Parameter(config.observation_mean, requires_grad=False)
        self.observation_std = nn.Parameter(config.observation_std + 1.e-6, requires_grad=False)
        self.action_mean = nn.Parameter(config.action_mean, requires_grad=False)
        self.action_std = nn.Parameter(config.action_std + 1.e-6, requires_grad=False)
        self.use_value = config.use_value
        self.use_new_s = config.use_new_s

    def create_layers(self, config):
        # embedding layers
        self.observation_embed = nn.Sequential(
            nn.Linear(self.observation_dim, self.embedding_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.embedding_dim)
        )
        self.embed_ln = nn.LayerNorm(self.embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.embedding_dim))
        if self.anystep:
            self.horizon_embed = nn.Embedding(self.horizon+1, self.embedding_dim)
            self.return_mean = nn.Sequential(
                nn.LayerNorm(self.output_dim+self.embedding_dim),
                nn.Linear(self.output_dim+self.embedding_dim, 1)
            )
            self.return_logvar = nn.Sequential(
                nn.LayerNorm(self.output_dim+self.embedding_dim),
                nn.Linear(self.output_dim+self.embedding_dim, 1)
            )
        else:
            # decoder layers
            self.return_mean = nn.Sequential(
                nn.LayerNorm(self.output_dim),
                nn.Linear(self.output_dim, 1)
            )
            self.return_logvar = nn.Sequential(
                nn.LayerNorm(self.output_dim),
                nn.Linear(self.output_dim, 1)
            )
        
        super().create_layers(config)
    
    def pad_to_full_observation(self, x):
        x_view = x.view(-1, self.transition_dim, self.embedding_dim)
        return x_view, 0
    
    def embed_inputs(self, inputs):
        observations = (inputs['observations'] - self.observation_mean) / self.observation_std
        actions = (inputs['actions'] - self.action_mean) / self.action_std
        b, obs_t, *_ = observations.shape
        _, act_t, *_ = actions.shape
        t = obs_t + act_t
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        observation_embeddings = self.observation_embed(observations)
        action_embeddings = self.action_embed(actions)

        # [ B x T x embedding_dim ]
        embeddings = torch.stack([observation_embeddings, action_embeddings], dim=2).reshape((b, t, self.embedding_dim))
        embeddings = self.embed_ln(embeddings)

        # [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector

        if 'embedding_offset' in inputs:
            position_embeddings = position_embeddings + inputs['embedding_offset']
        return embeddings + position_embeddings

    def decode_outputs(self, outputs, inputs):
        preds = {}
        if self.use_new_s:
            return_outputs = outputs[:, ::2]
        else:
            return_outputs = outputs[:, 1::2] # [B, T, E]
        if self.anystep:
            return_outputs = return_outputs.unsqueeze(2).expand(-1, -1, self.horizon, -1) # [B, T, H, E]
            horizon_inputs = torch.arange(1, self.horizon+1).expand(list(return_outputs.shape[:2])+[-1]).to(return_outputs.device) # [B, T, H]
            horizon_step_emb = self.horizon_embed(horizon_inputs) # [B, T, H, E] 
            return_outputs = torch.cat([return_outputs, horizon_step_emb], dim=-1) # [B, T, H, 2*E]
            
        preds['mean'] = self.return_mean(return_outputs) # [B, T, 1] or [B, T, H, 1]
        preds['logvar'] = self.return_logvar(return_outputs) # [B, T, 1] or [B, T, H, 1]
        
        return preds
    
    def estimate_return(self, inputs):
        outputs = self.process(inputs)
        preds = {}

        return_outputs = outputs[:, ::2]
        horizon_step_emb = self.horizon_embed(inputs['horizon_timesteps']) # [1, T, E]
        return_outputs = torch.cat([return_outputs, horizon_step_emb], dim=-1) # [1, T, 2*E]
        
        preds['mean'] = self.return_mean(return_outputs) # [1, T, 1]
        preds['logvar'] = self.return_logvar(return_outputs) # [1, T, 1]

        return preds

    def compute_loss(self, outputs, inputs, targets, mask=None):
        loss_dict = {}
        
        # Compute Loss Attenuation 
        if self.use_value:
            y, mu, var = targets['values'], outputs['mean'], torch.exp(outputs['logvar']) # [B, T, 1]
        elif self.anystep:
            y, mu, var = targets['anystep_returns'], outputs['mean'], torch.exp(outputs['logvar']) # [B, T, H, 1]
        else:
            y, mu, var = targets['returns'], outputs['mean'], torch.exp(outputs['logvar']) # [B, T, 1]
            
        std = torch.sqrt(var) 
        nll = (y - mu)**2 / (2 * torch.square(std)) + (1/2) * torch.log(torch.square(std))
        loss_dict['nll_loss'] = nll[mask[:, :-1]].mean()
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate_uncertainty(self, inputs):
        outputs = self.forward(inputs)
        return torch.exp(outputs['logvar'])
    
class EnsembleReturnGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.ensemble_size = config.ensemble_size
        self.create_layers(config)
        
    def create_layers(self, config):
        self.return_gpts = nn.ModuleList([ReturnGPT(config) for _ in range(self.ensemble_size)])
        
    def configure_optimizers(self, train_config):
        optimizers = []
        for i in range(self.ensemble_size):
            optimizers.append(self.return_gpts[i].configure_optimizers(train_config))
        return optimizers
    
    def forward(self, inputs):
        outputs = []
        for i in range(self.ensemble_size):
            outputs.append(self.return_gpts[i](inputs))
        return outputs
    
    def compute_loss(self, outputs, inputs, targets, mask=None): # mask [B, E, T, 1] or [B, E, T, H, 1]
        losses = []
        for i in range(self.ensemble_size):
            losses.append(self.return_gpts[i].compute_loss(outputs[i], inputs, targets, mask[:, i]))
        return losses
    
    @torch.no_grad()
    def evaluate_uncertainty(self, inputs):
        outputs = {'mean': [], 'var': []}
        for i in range(self.ensemble_size):
            output = self.return_gpts[i](inputs)
            outputs['mean'].append(output['mean'])
            outputs['var'].append(torch.exp(output['logvar']))
        return_mean = torch.stack(outputs['mean']) # [N, B, T, 1]
        return_var = torch.stack(outputs['var']) # [N, B, T, 1]
        
        uncertainty = (return_var + return_mean**2 - return_mean.mean(0).repeat(self.ensemble_size,1,1,1)**2).mean(0) # [B, T, 1]
        return uncertainty
    
class UNREST(GPT):
    def __init__(self, config):
        if not hasattr(config, 'output_dim'):
            config.output_dim = config.n_embd
        config.mask_values = False
        
        self.action_tanh = hasattr(config, 'action_tanh') and config.action_tanh
        self.horizon = config.horizon
        self.max_ep_length = config.max_ep_length
        self.embd_time = config.embd_time
        
        self.global_conditioned = config.global_conditioned
        if self.global_conditioned:
            self.n_global_condition_return_token = config.n_global_condition_return_token
            self.global_discretizer = ReturnUniformDiscretizer(config.global_return_min.to(config.device), config.global_return_max.to(config.device), config.n_global_condition_return_token, True)

        super().__init__(config)
        
        self.observation_mean = nn.Parameter(config.observation_mean, requires_grad=False)
        self.observation_std = nn.Parameter(config.observation_std + 1.e-6, requires_grad=False)
        self.action_mean = nn.Parameter(config.action_mean, requires_grad=False)
        self.action_std = nn.Parameter(config.action_std + 1.e-6, requires_grad=False)
        self.horizon_return_mean = nn.Parameter(config.horizon_return_mean, requires_grad=False)
        self.horizon_return_std = nn.Parameter(config.horizon_return_std + 1.e-6, requires_grad=False)
    
    def create_layers(self, config):
        ### embedding layers
        self.return_embed = nn.Sequential(
            nn.Linear(1, self.embedding_dim)
        )
        self.observation_embed = nn.Sequential(
            nn.Linear(self.observation_dim, self.embedding_dim)
        )
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, self.embedding_dim)
        )
        self.horizon_embed = nn.Embedding(self.horizon+1, self.embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.embedding_dim))
        self.embed_ln = nn.LayerNorm(self.embedding_dim)

        ### decoder layers
        if self.global_conditioned:
            self.action_decoder = nn.Sequential(
                nn.LayerNorm(self.output_dim+self.n_global_condition_return_token),
                nn.Linear(self.output_dim+self.n_global_condition_return_token, self.action_dim)
            )
        else:
            self.action_decoder = nn.Sequential(
                nn.LayerNorm(self.output_dim),
                nn.Linear(self.output_dim, self.action_dim)
            )
        super().create_layers(config)
    
    def pad_to_full_observation(self, x):
        x_view = x.view(-1, self.transition_dim, self.embedding_dim)
        return x_view, 0
    
    def embed_inputs(self, inputs):
        returns = (inputs['horizon_returns'] - self.horizon_return_mean) / self.horizon_return_std
        observations = (inputs['observations'] - self.observation_mean) / self.observation_std
        actions = (inputs['actions'] - self.action_mean) / self.action_std
        horizon_timesteps = inputs['horizon_timesteps']
        
        b, R_t, *_ = returns.shape
        _, obs_t, *_ = observations.shape
        _, act_t, *_ = actions.shape
        _, horizon_t, *_ = horizon_timesteps.shape
        t = R_t + obs_t + act_t
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        observation_embeddings = self.observation_embed(observations)
        action_embeddings = self.action_embed(actions)
        if 'certain_mask' in inputs:
            certain_masks = inputs['certain_mask']
            certain_indices = certain_masks != 0
            uncertain_indices = certain_masks == 0
            certain_returns = returns[certain_indices]
            certain_horizon_timesteps = horizon_timesteps[certain_indices]
            uncertain_horizon_timesteps = horizon_timesteps[uncertain_indices]
            
            certain_return_embeddings = self.return_embed(certain_returns)
            uncertain_return_embeddings = self.horizon_embed(uncertain_horizon_timesteps)
            if self.embd_time:
                certain_horizon_embeddings = self.horizon_embed(certain_horizon_timesteps)
                certain_return_embeddings = certain_return_embeddings + certain_horizon_embeddings
            
            return_embeddings = torch.zeros([b] + list(certain_return_embeddings.shape[1:])).to(certain_return_embeddings.device)
            return_embeddings[certain_indices] = certain_return_embeddings
            return_embeddings[uncertain_indices] = uncertain_return_embeddings
        else:
            return_embeddings = self.return_embed(returns)
            if self.embd_time:
                horizon_embeddings = self.horizon_embed(horizon_timesteps)
                return_embeddings = return_embeddings + horizon_embeddings
        
        ## [ B x T x embedding_dim ]
        embeddings = torch.stack([return_embeddings, observation_embeddings, action_embeddings], dim=2).reshape((b, t, self.embedding_dim))
        embeddings = self.embed_ln(embeddings)
        
        ## [ 1 x T x embedding_dim ]
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        if 'embedding_offset' in inputs:
            position_embeddings = position_embeddings + inputs['embedding_offset']
        return embeddings + position_embeddings
    
    def decode_outputs(self, outputs, inputs):
        action_pred_outputs = outputs[:, 1::3]
        if self.global_conditioned:
            global_return = self.global_discretizer.discretize(inputs['returns']) # [B, T, N]
            action_pred_outputs = torch.cat([action_pred_outputs, global_return], dim=-1)

        preds = {}

        if self.action_tanh:
            preds['actions'] = self.action_decoder(action_pred_outputs).tanh()
        else:
            preds['actions'] = self.action_std * self.action_decoder(action_pred_outputs) + self.action_mean
        return preds

    def compute_loss(self, outputs, inputs, targets, mask=None):
        loss_dict = {}
        if self.action_tanh:
            target_actions = targets['actions'].clamp(-0.999, 0.999)
        else:
            target_actions = targets['actions']
        action_error = F.mse_loss(outputs['actions'], target_actions, reduction='none')
        action_loss = torch.sum(action_error / (self.action_std ** 2), dim=-1, keepdims=True)
        loss_dict['action_loss'] = action_loss[mask[:, :-1]].mean()
        return loss_dict