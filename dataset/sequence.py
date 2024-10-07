import numpy as np
import torch

from .data_utils import load_environment, qlearning_dataset_with_timeouts, qlearning_dataset, to_torch

def segment(observations, terminals, max_path_length):
    """
        segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    curr_len = 0
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        curr_len += 1
        if term.squeeze() or (curr_len >= max_path_length):
            trajectories.append([])
            curr_len = 0

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1

    return trajectories_pad, early_termination, path_lengths

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

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env=None, sequence_length=250, step=10, discount=0.99, max_path_length=1000, target_offset=1, penalty=None, device='cuda:0', dataset=None, timeouts=True, esper=False, **kwargs):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device

        self.target_offset = target_offset

        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        if timeouts:
            dataset = qlearning_dataset_with_timeouts(env=env.unwrapped if env else env, dataset=dataset, terminate_on_end=True, esper=esper)
        else:
            dataset = qlearning_dataset(env=env.unwrapped if env else env, dataset=dataset, esper=esper)
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
        if esper:
            self.esper_segmented, *_ = segment(dataset['esper_returns'], terminals, max_path_length)
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
        if esper:
            self.returns_segmented = self.esper_segmented

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

        ## get valid indices
        indices = []
        for path_ind, length in enumerate(self.path_lengths):
            if length < 300: # filter out short experiences
                continue
            end = length
            for i in range(end):
                indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.n_trajectories = n_trajectories
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.next_observations_segmented = np.concatenate([
            self.next_observations_segmented,
            np.zeros((n_trajectories, sequence_length-1, self.observation_dim)),
        ], axis=1)
        self.terminals_segmented = np.concatenate([
            self.terminals_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.returns_segmented = np.concatenate([
            self.returns_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.orig_returns_segmented = np.concatenate([
            self.orig_returns_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.termination_flags_orig = self.termination_flags
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool),
        ], axis=1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]

        ## [ (sequence_length / skip) x observation_dim]
        joined = to_torch(joined, device='cpu', dtype=torch.float).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0
        # TODO mask is problematic if not predicting terminals

        ## flatten everything
        joined = joined.view(-1)
        mask = mask.view(-1)

        X = {
            'transitions': joined[:-1],
        }
        Y = {
            'transitions': joined[self.target_offset:],
        }
        mask = mask[self.target_offset:]

        return X, Y, mask

    def get_stats(self):
        unfilt_diffs = self.observations_raw[1:] - self.observations_raw[:-1]
        diffs = unfilt_diffs[~self.terminals_raw[:-1, 0].astype(bool)]
        return {
            'observation_mean': self.observations_raw.mean(axis=0),
            'observation_std': self.observations_raw.std(axis=0),
            'action_mean': self.actions_raw.mean(axis=0),
            'action_std': self.actions_raw.std(axis=0),
            'reward_mean': self.rewards_raw.mean(axis=0),
            'reward_std': self.rewards_raw.std(axis=0),
            'value_mean': self.values_raw.mean(axis=0),
            'value_std': self.values_raw.std(axis=0),
            'observation_diff_mean': diffs.mean(axis=0),
            'observation_diff_std': diffs.std(axis=0),
            'return_mean': self.returns_raw.mean(axis=0),
            'return_std': self.returns_raw.std(axis=0),
            'orig_return_mean': self.orig_returns_raw.mean(axis=0),
            'orig_return_std': self.orig_returns_raw.std(axis=0),

            'observation_max': self.observations_raw.max(axis=0),
            'observation_min': self.observations_raw.min(axis=0),
            'action_max': self.actions_raw.max(axis=0),
            'action_min': self.actions_raw.min(axis=0),
            'reward_max': self.rewards_raw.max(axis=0),
            'reward_min': self.rewards_raw.min(axis=0),
            'value_max': self.values_raw.max(axis=0),
            'value_min': self.values_raw.min(axis=0),
            'observation_diff_max': diffs.max(axis=0),
            'observation_diff_min': diffs.min(axis=0),
            'return_max': self.returns_raw.max(axis=0),
            'return_min': self.returns_raw.min(axis=0),
            'orig_return_max': self.orig_returns_raw.max(axis=0),
            'orig_return_min': self.orig_returns_raw.min(axis=0),
        }

    def get_max_return(self):
        return self.returns_segmented[:, 0, 0].max()

    def get_min_return(self):
        return self.returns_segmented[:, 0, 0].min()
    
    def get_mean_return(self):
        return self.returns_segmented[:, 0, 0].mean()
    
    def get_orig_max_return(self):
        return self.orig_returns_segmented[:, 0, 0].max()

    def get_orig_min_return(self):
        return self.orig_returns_segmented[:, 0, 0].min()
    
    def get_orig_mean_return(self):
        return self.orig_returns_segmented[:, 0, 0].mean()

class HorizonSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, env=None, sequence_length=250, step=10, discount=1., max_path_length=1000, target_offset=1, penalty=None, device='cuda:0', dataset=None, timeouts=True, horizon=100, anystep=False, indice_path=None, certain_only=False, uncertain_only=False, **kwargs):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.horizon = horizon
        self.device = device
        self.anystep = anystep
        self.certain_only = certain_only
        self.uncertain_only = uncertain_only
        self.traj_indices = np.load(indice_path).astype(np.int32) if indice_path is not None else None

        self.target_offset = target_offset

        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        if timeouts:
            dataset = qlearning_dataset_with_timeouts(env=env.unwrapped if env else env, dataset=dataset, terminate_on_end=True)
        else:
            dataset = qlearning_dataset(env=env.unwrapped if env else env, dataset=dataset)
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
        self.discounts = (discount ** np.arange(self.horizon))[:,None]
        self.orig_discounts = np.ones_like(self.discounts)

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape)
        self.returns_segmented = np.zeros(self.rewards_segmented.shape)

        for t in range(max_path_length-self.horizon-1):
            ## [ n_paths x 1 ]
            V = self.rewards_segmented[:,t+1:t+1+self.horizon].sum(axis=1)
            self.values_segmented[:,t] = V

            R = self.rewards_segmented[:,t:t+self.horizon].sum(axis=1)
            self.returns_segmented[:,t] = R
        if self.anystep:
            cumsum_rewards = np.concatenate((np.zeros((self.rewards_segmented.shape[0],1)), np.cumsum(self.rewards_segmented.squeeze(), axis=1)), axis=1)
            indices = np.arange(self.max_path_length).reshape(-1, 1) + np.arange(self.horizon) + 1
            clipped_indices = np.clip(indices, None, cumsum_rewards.shape[1] - 1)
            self.anystep_R = (cumsum_rewards[:, clipped_indices] - cumsum_rewards[:, :self.max_path_length].reshape(self.rewards_segmented.shape[0], -1, 1))[..., None]

        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        self.values_raw = values_raw[values_mask, None]
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)

        returns_raw = self.returns_segmented.squeeze(axis=-1).reshape(-1)
        returns_mask = ~self.termination_flags.reshape(-1)
        self.returns_raw = returns_raw[returns_mask, None]

        ## get valid indices
        indices = []
        certain_masks = []
        for path_ind, length in enumerate(self.path_lengths):
            if length < 300 or length < self.horizon: # filter out short experiences
                continue
            
            if self.traj_indices is not None:
                nonzero_indices = list(np.nonzero(self.traj_indices[path_ind])[0])
                zero_indices = list(np.where(self.traj_indices[path_ind][:length] == 0)[0])
                selected_indices = [zero_indices, nonzero_indices]
                
                for k, selected_indice in enumerate(selected_indices):
                    if k == 0 and self.certain_only:
                        continue
                    if k == 1 and self.uncertain_only:
                        continue
                    if len(selected_indice) == 0:
                        continue
                    
                    pprev_idx = prev_idx = selected_indice[0]
                    for idx in selected_indice[1:]:
                        if idx != prev_idx + 1:
                            if prev_idx - pprev_idx >= sequence_length - 1:
                                for i in range(pprev_idx, prev_idx-sequence_length+2):
                                    indices.append((path_ind, i, i+sequence_length))
                                    certain_masks.append(k)
                            pprev_idx = idx
                        prev_idx = idx
                    if pprev_idx != prev_idx and prev_idx - pprev_idx >= sequence_length - 1:
                        for i in range(pprev_idx, prev_idx-sequence_length+2):
                            indices.append((path_ind, i, i+sequence_length))
                            certain_masks.append(k)
            else:
                # end = length - sequence_length + 1
                end = length - self.horizon - sequence_length
                for i in range(end):
                    indices.append((path_ind, i, i+sequence_length))

        self.indices = np.array(indices)
        self.certain_masks = np.array(certain_masks)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.n_trajectories = n_trajectories
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim)),
        ], axis=1)
        self.next_observations_segmented = np.concatenate([
            self.next_observations_segmented,
            np.zeros((n_trajectories, sequence_length-1, self.observation_dim)),
        ], axis=1)
        self.terminals_segmented = np.concatenate([
            self.terminals_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.returns_segmented = np.concatenate([
            self.returns_segmented,
            np.zeros((n_trajectories, sequence_length-1, 1)),
        ], axis=1)
        self.termination_flags_orig = self.termination_flags
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool),
        ], axis=1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_ind, start_ind, end_ind = self.indices[idx]

        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]

        ## [ (sequence_length / skip) x observation_dim]
        joined = to_torch(joined, device='cpu', dtype=torch.float).contiguous()

        ## don't compute loss for parts of the prediction that extend
        ## beyond the max path length
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones(joined.shape, dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0
        # TODO mask is problematic if not predicting terminals

        ## flatten everything
        joined = joined.view(-1)
        mask = mask.view(-1)

        X = {
            'transitions': joined[:-1],
        }
        Y = {
            'transitions': joined[self.target_offset:],
        }
        mask = mask[self.target_offset:]

        return X, Y, mask

    def get_stats(self):
        unfilt_diffs = self.observations_raw[1:] - self.observations_raw[:-1]
        diffs = unfilt_diffs[~self.terminals_raw[:-1, 0].astype(bool)]
        return {
            'observation_mean': self.observations_raw.mean(axis=0),
            'observation_std': self.observations_raw.std(axis=0),
            'action_mean': self.actions_raw.mean(axis=0),
            'action_std': self.actions_raw.std(axis=0),
            'reward_mean': self.rewards_raw.mean(axis=0),
            'reward_std': self.rewards_raw.std(axis=0),
            'value_mean': self.values_raw.mean(axis=0),
            'value_std': self.values_raw.std(axis=0),
            'observation_diff_mean': diffs.mean(axis=0),
            'observation_diff_std': diffs.std(axis=0),
            'return_mean': self.returns_raw.mean(axis=0),
            'return_std': self.returns_raw.std(axis=0),

            'observation_max': self.observations_raw.max(axis=0),
            'observation_min': self.observations_raw.min(axis=0),
            'action_max': self.actions_raw.max(axis=0),
            'action_min': self.actions_raw.min(axis=0),
            'reward_max': self.rewards_raw.max(axis=0),
            'reward_min': self.rewards_raw.min(axis=0),
            'value_max': self.values_raw.max(axis=0),
            'value_min': self.values_raw.min(axis=0),
            'observation_diff_max': diffs.max(axis=0),
            'observation_diff_min': diffs.min(axis=0),
            'return_max': self.returns_raw.max(axis=0),
            'return_min': self.returns_raw.min(axis=0),
        }

    def get_max_return(self):
        return self.returns_segmented[:, 0, 0].max()

    def get_min_return(self):
        return self.returns_segmented[:, 0, 0].min()
    
    def get_mean_return(self):
        return self.returns_segmented[:, 0, 0].mean()