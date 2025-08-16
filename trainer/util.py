from torch.utils.data import DataLoader, Dataset
import torch
import math
import yaml

with open('config.yaml', "r") as f:
    CONFIG = yaml.safe_load(f)
    f.close()

def get_dataloader(ds: Dataset):
    dataloader = DataLoader(
        dataset=ds,
        batch_size=CONFIG['train_config']['batchsize'],
        num_workers=4,
        prefetch_factor=4
    )
    return dataloader

    
def get_lr(step, total_steps):
    warmup_steps = CONFIG['train_config']['warmup_steps']
    max_lr = CONFIG['train_config']['max_lr']
    min_lr = CONFIG['train_config']['min_lr']
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))