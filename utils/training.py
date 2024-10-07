import math
import torch
from torch.utils.data.dataloader import DataLoader

def to(xs, device):
    res = []
    for x in xs:
        if isinstance(x, dict):
            for k in x:
                x[k] = x[k].to(device)
            res.append(x)
        else:
            res.append(x.to(device))
    return res

def compute_total_loss(losses):
    total_loss = torch.stack([loss[1] for loss in losses.items()]).sum()
    return total_loss

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.logger = config.logger

        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.optimizer = None
        self.step = 0

    def get_optimizer(self, model):
        if self.optimizer is None:
            self.logger.info(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, model, dataset, n_epochs=1, log_freq=100):

        config = self.config
        optimizer = self.get_optimizer(model)
        model.train(True)

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        for _ in range(n_epochs):

            losses = []
            for it, batch in enumerate(loader):

                batch = to(batch, self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    outputs = model(batch[0])
                    loss_dict = model.compute_loss(outputs, *batch)
                    total_loss = loss_dict['total_loss'] = compute_total_loss(loss_dict)
                    losses.append(loss_dict['total_loss'].item())

                # backprop and update the parameters
                model.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                mask = batch[-1]
                self.n_tokens += mask.sum() # number of tokens processed this step
                # decay the learning rate based on our progress
                if config.lr_decay:
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        lr_mult = 1.0
                    lr = lr_mult * config.learning_rate

                # report progress
                if it % log_freq == 0:
                    for name, loss in loss_dict.items():
                        self.logger.info(f'[ utils/training ] epoch {self.n_epochs} [ {it} / {len(loader)} ] ' \
                        f'train loss {total_loss.item():.5} | lr {lr:.3} | lr_mult: {lr_mult:.4}')
                self.step += 1

            self.n_epochs += 1

class EnsembleTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.logger = config.logger
        self.ensemble_size = config.ensemble_size

        self.n_epochs = 0
        self.n_tokens = [0 for _ in range(self.ensemble_size)] # counter used for learning rate decay
        self.optimizers = None
        self.step = 0
    
    def get_optimizer(self, model):
        if self.optimizers is None:
            self.logger.info(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizers = model.configure_optimizers(self.config)
        return self.optimizers

    def train(self, model, dataset, n_epochs=1, log_freq=100):
        config = self.config
        optimizers = self.get_optimizer(model)
        model.train(True)
        
        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        
        for _ in range(n_epochs):
            for it, batch in enumerate(loader):
                batch = to(batch, self.device)
                
                with torch.set_grad_enabled(True):
                    outputs = model(batch[0])
                    losses = model.compute_loss(outputs, *batch)
                    for i in range(len(losses)):
                        losses[i]['total_loss'] = compute_total_loss(losses[i])
                
                for i in range(self.ensemble_size):
                    total_loss = losses[i]['total_loss']
                    model.return_gpts[i].zero_grad()
                    losses[i]['total_loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.return_gpts[i].parameters(), config.grad_norm_clip)
                    optimizers[i].step()
                    
                    mask = batch[-1][:, i]
                    self.n_tokens[i] += mask.sum()
                    
                    if config.lr_decay:
                        if self.n_tokens[i] < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.n_tokens[i]) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.n_tokens[i] - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizers[i].param_groups:
                            param_group['lr'] = lr
                    else:
                        if self.n_tokens[i] < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.n_tokens[i]) / float(max(1, config.warmup_tokens))
                        else:
                            lr_mult = 1.0
                        lr = lr_mult * config.learning_rate
                    
                    # report progress
                    if it % log_freq == 0:
                        for name, loss in losses[i].items():
                            self.logger.info(f'[ utils/training ] epoch {self.n_epochs} [ {it} / {len(loader)} ] model {i} ' \
                            f'train loss {total_loss.item():.5} | lr {lr:.3} | lr_mult: {lr_mult:.4}')
                
                self.step += 1
            
            self.n_epochs += 1