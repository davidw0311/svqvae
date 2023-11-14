import torch
from models.QVaLT import QVaLT
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torch.nn as nn
import os, shutil, datetime
import logging
from data.stats import data
import numpy as np
from tqdm import tqdm

def get_wbc_dataset(type):
    
    mean = data['train']['mean']
    std = data['train']['std']
    
    if type == 'val':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    
    path = data['train']['paths'][type]

    dataset = ImageFolder(root=path, transform=transform)
    return dataset

if __name__ == '__main__':
    
    training_dataset = get_wbc_dataset('wbc_100')
    targets_tensor = torch.tensor(training_dataset.targets)
    class_sample_count = torch.tensor([(targets_tensor == t).sum() for t in torch.unique(targets_tensor, sorted=True)])
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t] for t in training_dataset.targets])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(training_dataset, batch_size=32, sampler=sampler, num_workers=2)

    seen = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for step, (image_batch, labels_batch) in tqdm(enumerate(train_loader)):
        for i in labels_batch:
            seen[i.item()] += 1

    print(seen)