import torch
from torch.distributions.binomial import Binomial
import numpy as np

from .data_utils import to_torch, qlearning_dataset, qlearning_dataset_with_timeouts, load_environment
from .sequence import SequenceDataset, HorizonSequenceDataset, segment

def calculate_cumulative_rewards(rewards, indices):
    row_idx = np.arange(rewards.shape[0])[:, None]
    col_idx = np.arange(rewards.shape[1])
    cumsum_rewards = np.concatenate((np.zeros((rewards.shape[0], 1)), np.cumsum(rewards.squeeze(), axis=1)), axis=1)
    end_col_idx = np.clip(col_idx+indices, None, cumsum_rewards.shape[1]-1)
    cumulative_rewards = cumsum_rewards[row_idx, end_col_idx] - cumsum_rewards[row_idx, col_idx]
    return cumulative_rewards[..., None]

class SegTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, env=None, sequence_length=250, step=10, discount=0.99, max_path_length=1000, target_offset=1, penalty=None, device='cuda:0', dataset=None, timeouts=True, **kwargs):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device

        self.target_offset = target_offset

        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        if timeouts:
            dataset = qlearning_dataset_with_timeouts(env=env.unwrapped if env else env, dataset=dataset, terminate_on_end=True, esper=False)
        else:
            dataset = qlearning_dataset(env=env.unwrapped if env else env, dataset=dataset, esper=False)
        # print('✓')

        observations = dataset['observations']
        actions = dataset['actions']
        next_observations = dataset['next_observations']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        realterminals = dataset['realterminals']

        self.observations_raw = observations
        self.actions_raw = actions
        self.next_observations_raw = next_observations
        self.joined_raw = np.concatenate([observations, actions], axis=-1)
        self.rewards_raw = rewards
        self.terminals_raw = terminals

        ## terminal penalty
        if penalty is not None:
            terminal_mask = realterminals.squeeze()
            self.rewards_raw[terminal_mask] = penalty

        ## segment
        print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        self.next_observations_segmented, *_ = segment(self.next_observations_raw, terminals, max_path_length)
        self.terminals_segmented, *_ = segment(self.terminals_raw, terminals, max_path_length)
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length)
        # print('✓')
        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]
        self.orig_discounts = np.ones_like(self.discounts)
        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)
        self.returns_segmented = np.zeros(self.rewards_segmented.shape)
        self.orig_returns_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
            self.values_segmented[:,t] = V

            R = (self.rewards_segmented[:,t:] * self.discounts[:self.max_path_length-t]).sum(axis=1)
            self.returns_segmented[:,t] = R
            
            orig_R = (self.rewards_segmented[:,t:] * self.orig_discounts[:self.max_path_length-t]).sum(axis=1)
            self.orig_returns_segmented[:,t] = orig_R

        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)

        returns_raw = self.returns_segmented.squeeze(axis=-1).reshape(-1)
        orig_returns_raw = self.orig_returns_segmented.squeeze(axis=-1).reshape(-1)
        returns_mask = ~self.termination_flags.reshape(-1)
        self.returns_raw = returns_raw[returns_mask, None]
        self.orig_returns_raw = orig_returns_raw[returns_mask, None]

        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]
        
    def __len__(self):
        return len(self.path_lengths)

    def __getitem__(self, idx):
        joined = self.joined_segmented[idx]
        terminations = self.termination_flags[idx] # [T]
        next_observations = self.next_observations_segmented[idx]
        terminals = self.terminals_segmented[idx]
        returns = self.returns_segmented[idx]
        orig_returns = self.orig_returns_segmented[idx]
        path_length = self.path_lengths[idx]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        rewards = joined[:, -2:-1]
        values = joined[:, -1:]

        returns = to_torch(returns, device='cpu').contiguous()
        orig_returns = to_torch(orig_returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(1) # [T, 1]

        X = {
            'observations': observations[:],
            'actions': actions[:],
            'rewards': rewards[:],
            'values': values[:],
            'terminals': terminals[:],
            'returns': returns[:],
            'orig_returns': orig_returns[:],
            'mask': mask,
            'traj_length': path_length
        }

        return X

class TrajectoryDataset(SequenceDataset):
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step] # [T]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]
        orig_returns = self.orig_returns_segmented[path_ind, start_ind:end_ind:self.step]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        rewards = joined[:, -2:-1]
        values = joined[:, -1:]

        returns = to_torch(returns, device='cpu').contiguous()
        orig_returns = to_torch(orig_returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(1) # [T, 1]
        mask[traj_inds > self.max_path_length - self.step] = 0

        X = {
            'observations': observations[:-1],
            'next_observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'orig_returns': orig_returns[:-1],
            'traj_indices': path_ind,
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'orig_returns': orig_returns[:-1]
        }
        return X, Y, mask
    
class EnsembleTrajectoryDataset(TrajectoryDataset):
    def __init__(self, *args, ensemble_size=5, mask_prob=0.6, **kwargs):
        self.ensemble_size = ensemble_size
        self.mask_prob = mask_prob
        
        super().__init__(*args, **kwargs)
        
        self.termination_flags = self.termination_flags[None, :, :].repeat(self.ensemble_size, axis=0)
        binomial_dist = Binomial(torch.ones(list(self.termination_flags.shape)), mask_prob)
        self.ensemble_mask = binomial_dist.sample().contiguous().bool()
    
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[:, path_ind, start_ind:end_ind:self.step] # [E, T]
        ensemble_mask = self.ensemble_mask[:, path_ind, start_ind:end_ind:self.step].unsqueeze(2) # [E, T, 1]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        rewards = joined[:, -2:-1]
        values = joined[:, -1:]

        returns = to_torch(returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(2) # [E, T, 1] 
        mask[:, traj_inds > self.max_path_length - self.step] = 0
        mask.mul_(ensemble_mask) # [E, T, 1]

        X = {
            'observations': observations[:-1],
            'next_observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'traj_indices': path_ind,
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
        }
        return X, Y, mask

class HorizonTrajectoryDataset(HorizonSequenceDataset):
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step] # [T]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]
        if self.anystep:
            anystep_returns = self.anystep_R[path_ind, start_ind:end_ind:self.step]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]

        returns = to_torch(returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(1) # [T, 1]
        mask[traj_inds > self.max_path_length - self.step] = 0

        X = {
            'observations': observations[:-1],
            # 'next_observations': observations[1:],
            'actions': actions[:-1],
            # 'rewards': rewards[:-1],
            # 'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'traj_indices': path_ind,
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            # 'rewards': rewards[:-1],
            # 'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
        }
        
        if self.anystep:
            Y['anystep_returns'] = anystep_returns[:-1]
            mask = mask.unsqueeze(1).expand(-1, self.horizon, -1)
        
        return X, Y, mask

class EnsembleHorizonTrajectoryDataset(HorizonTrajectoryDataset):
    def __init__(self, *args, ensemble_size=5, mask_prob=0.6, **kwargs):
        self.ensemble_size = ensemble_size
        self.mask_prob = mask_prob
        
        super().__init__(*args, **kwargs)
        
        self.termination_flags = self.termination_flags[None, :, :].repeat(self.ensemble_size, axis=0)
        binomial_dist = Binomial(torch.ones(list(self.termination_flags.shape)), mask_prob)
        self.ensemble_mask = binomial_dist.sample().contiguous().bool()
        
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[:, path_ind, start_ind:end_ind:self.step] # [E, T]
        ensemble_mask = self.ensemble_mask[:, path_ind, start_ind:end_ind:self.step].unsqueeze(2) # [E, T, 1]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]
        if self.anystep:
            anystep_returns = self.anystep_R[path_ind, start_ind:end_ind:self.step]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]

        returns = to_torch(returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(2) # [E, T, 1] 
        mask[:, traj_inds > self.max_path_length - self.step] = 0
        mask.mul_(ensemble_mask) # [E, T, 1]
        
        X = {
            'observations': observations[:-1],
            'actions': actions[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'traj_indices': path_ind,
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
        }
        
        if self.anystep:
            Y['anystep_returns'] = anystep_returns[:-1]
            mask = mask.unsqueeze(2).expand(-1, -1, self.horizon, -1) # [E, T, H, 1] 
        
        return X, Y, mask
    
class FixedHorizonTrajectoryDataset(HorizonTrajectoryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.horizon_discounts = (self.discount ** np.arange(self.horizon))[:,None]
        self.horizon_return_segmented = np.zeros(self.rewards_segmented.shape)
        rewards_splt = torch.split(torch.tensor(self.rewards_segmented), self.horizon, 1)
        
        for t in range(self.max_path_length):
            group = t // self.horizon
            group_step = t % self.horizon
            R = (rewards_splt[group][:, group_step:] * self.horizon_discounts[:self.horizon-group_step]).sum(axis=1)
            self.horizon_return_segmented[:, t] = R
        
        horizon_return_raw = self.horizon_return_segmented.squeeze(axis=-1).reshape(-1)
        returns_mask = ~self.termination_flags_orig.reshape(-1)
        self.horizon_return_raw = horizon_return_raw[returns_mask, None]
        
        self.horizon_return_segmented = np.concatenate([
            self.horizon_return_segmented,
            np.zeros((self.n_trajectories, self.sequence_length-1, 1)),
        ], axis=1)
        
    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step] # [T]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]
        
        if self.anystep:
            step_left = self.path_lengths[path_ind] - start_ind
            if step_left >= self.sequence_length + self.horizon - 1:
                choices = list(np.arange(1, self.horizon+1))
            elif step_left < self.sequence_length + self.horizon - 1 and step_left >= self.horizon:
                choices = list(np.arange(self.sequence_length, self.horizon+1)) + list(np.arange(1, step_left - self.horizon + 1))
            else:
                choices = list(np.arange(self.sequence_length, step_left+1))
            start = np.random.choice(choices)
            return_spans = (np.arange(start, start-self.sequence_length, -self.step) - 1) % self.horizon
            horizon_returns = self.anystep_R[path_ind][range(start_ind, end_ind, self.step), return_spans]
            return_spans = return_spans + 1
        else:
            horizon_returns = self.horizon_return_segmented[path_ind, start_ind:end_ind:self.step]
            return_spans = np.min([self.horizon - np.arange(start_ind, end_ind, self.step) % self.horizon, self.path_lengths[path_ind] - np.arange(start_ind, end_ind, self.step)], axis=0).clip(0)

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        rewards = joined[:, -2:-1]
        values = joined[:, -1:]

        returns = to_torch(returns, device='cpu').contiguous()
        horizon_returns = to_torch(horizon_returns, device='cpu').contiguous()

        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(1) # [T, 1]
        mask[traj_inds > self.max_path_length - self.step] = 0

        X = {
            'observations': observations[:-1],
            'next_observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'horizon_returns': horizon_returns[:-1],
            'traj_indices': path_ind,
            'timesteps': to_torch(np.arange(start_ind, end_ind, self.step), dtype=torch.int32, device='cpu').contiguous()[:-1], # [T]
            'horizon_timesteps': to_torch(return_spans, dtype=torch.int32, device='cpu').contiguous()[:-1] # [T]
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'horizon_returns': horizon_returns[:-1],
        }
        return X, Y, mask
    
    def get_stats(self):
        stats_dict = super().get_stats()
        stats_dict['horizon_return_mean'] = self.horizon_return_raw.mean(axis=0)
        stats_dict['horizon_return_std'] = self.horizon_return_raw.std(axis=0)
        stats_dict['horizon_return_min'] = self.horizon_return_raw.min(axis=0)
        stats_dict['horizon_return_max'] = self.horizon_return_raw.max(axis=0)
        return stats_dict
    
    def get_horizon_max_return(self):
        return self.horizon_return_segmented[:, 0].max()

    def get_horizon_min_return(self):
        return self.horizon_return_segmented[:, 0].min()
    
    def get_horizon_mean_return(self):
        return self.horizon_return_segmented[:, 0].mean()
    
class IndicedHorizonDataset(HorizonTrajectoryDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.traj_indices is not None, "You need to specify indice_path!"
        assert self.discount == 1, "Currently only support discount == 1!"
        self.horizon_max = self.traj_indices.max()
        
        # calculate horizon returns using indices
        self.horizon_return_segmented = calculate_cumulative_rewards(self.rewards_segmented, self.traj_indices)
        horizon_return_raw = self.horizon_return_segmented.squeeze(axis=-1).reshape(-1)
        returns_mask = ~self.termination_flags_orig.reshape(-1)
        self.horizon_return_raw = horizon_return_raw[returns_mask, None]
        
        self.horizon_return_segmented = np.concatenate([
            self.horizon_return_segmented,
            np.zeros((self.n_trajectories, self.sequence_length-1, 1)),
        ], axis=1)
        self.traj_indices_segmented = np.concatenate([
            self.traj_indices,
            np.zeros((self.n_trajectories, self.sequence_length-1)),
        ], axis=1).astype(np.int32)
    
    def __getitem__(self, idx):
        certain_mask = self.certain_masks[idx]
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        terminations = self.termination_flags[path_ind, start_ind:end_ind:self.step] # [T]
        next_observations = self.next_observations_segmented[path_ind, start_ind:end_ind:self.step]
        terminals = self.terminals_segmented[path_ind, start_ind:end_ind:self.step]
        returns = self.returns_segmented[path_ind, start_ind:end_ind:self.step]
        
        if self.anystep and certain_mask:
            step_left = self.traj_indices_segmented[path_ind, start_ind]
            if step_left >= self.sequence_length + self.horizon - 1:
                choices = list(np.arange(1, self.horizon+1))
            elif step_left < self.sequence_length + self.horizon - 1 and step_left >= self.horizon:
                choices = list(np.arange(self.sequence_length, self.horizon+1)) + list(np.arange(1, step_left - self.horizon + 1))
            else:
                choices = list(np.arange(self.sequence_length, step_left+1))
            start = np.random.choice(choices)
            return_spans = (np.arange(start, start-self.sequence_length, -self.step) - 1) % self.horizon
            horizon_returns = self.anystep_R[path_ind][range(start_ind, end_ind, self.step), return_spans]
            return_spans = return_spans + 1
        else:
            horizon_returns = self.horizon_return_segmented[path_ind, start_ind:end_ind:self.step]
            return_spans = self.traj_indices_segmented[path_ind, start_ind:end_ind:self.step]

        joined = to_torch(joined, device='cpu').contiguous()
        observations = joined[:, :self.observation_dim]
        actions = joined[:, self.observation_dim:self.observation_dim+self.action_dim]
        rewards = joined[:, -2:-1]
        values = joined[:, -1:]

        returns = to_torch(returns, device='cpu').contiguous()
        horizon_returns = to_torch(horizon_returns, device='cpu').contiguous()
        next_observations = to_torch(next_observations, device='cpu').contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = ~to_torch(terminations, device='cpu').contiguous().bool().unsqueeze(1) # [T, 1]
        mask[traj_inds > self.max_path_length - self.step] = 0

        X = {
            'observations': observations[:-1],
            'next_observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'horizon_returns': horizon_returns[:-1],
            'traj_indices': path_ind,
            'certain_mask': certain_mask,
            'timesteps': to_torch(np.arange(start_ind, end_ind, self.step), dtype=torch.int32, device='cpu').contiguous()[:-1], # [T]
            'horizon_timesteps': to_torch(return_spans, device='cpu', dtype=torch.int32).contiguous()[:-1] # [T]
        }

        Y = {
            'observations': observations[1:],
            'actions': actions[:-1],
            'rewards': rewards[:-1],
            'values': values[:-1],
            'terminals': terminals[:-1],
            'returns': returns[:-1],
            'horizon_returns': horizon_returns[:-1],
        }
        return X, Y, mask
    
    def get_stats(self):
        stats_dict = super().get_stats()
        stats_dict['horizon_return_mean'] = self.horizon_return_raw.mean(axis=0)
        stats_dict['horizon_return_std'] = self.horizon_return_raw.std(axis=0)
        stats_dict['horizon_return_min'] = self.horizon_return_raw.min(axis=0)
        stats_dict['horizon_return_max'] = self.horizon_return_raw.max(axis=0)
        return stats_dict
    
    def get_horizon_max_return(self):
        return self.horizon_return_segmented[:, 0].max()

    def get_horizon_min_return(self):
        return self.horizon_return_segmented[:, 0].min()
    
    def get_horizon_mean_return(self):
        return self.horizon_return_segmented[:, 0].mean()