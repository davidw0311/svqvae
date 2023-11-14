import torch
from models.svqvae import SVQVAE
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import torch.nn as nn
import os, shutil, datetime
import logging
from data.stats import data
import numpy as np
import argparse
import json
import time
from types import SimpleNamespace
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def denormalize_img(img, mean, std):
    for t,m,s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    img = torch.clamp(img, 0,1)
    return img.permute((1,2,0))

class MaskDataset(Dataset):
    def __init__(self, data_root, mask_root, spatial_transform, image_transform):
        self.data_root = data_root
        self.mask_root = mask_root
        self.spatial_transform = spatial_transform
        self.image_transform = image_transform
        self.samples = []
        self.targets = []

        # Iterate over each class folder
        classes_folder = sorted([i for i in os.listdir(data_root) if not i.startswith('.')])
        for label_idx, class_folder in enumerate(classes_folder):
            if class_folder.startswith('.'):
                continue
            image_class_dir = os.path.join(data_root, class_folder)
            mask_class_dir = os.path.join(mask_root, class_folder)
            for image_name in os.listdir(image_class_dir):
                if image_name.startswith('.'):
                    continue
                self.samples.append((image_class_dir, mask_class_dir, image_name, label_idx))
                self.targets.append(label_idx)
                
    def __len__(self):
        return len(self.samples)

    
    def __getitem__(self, idx):
        image_class_dir, mask_class_dir, image_name, label = self.samples[idx]
        
        img_path = os.path.join(image_class_dir, image_name)
        mask_path = os.path.join(mask_class_dir, image_name)

        image = Image.open(img_path).convert("RGB")

        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("RGB")
        else:
            mask = Image.new("RGB", image.size, color=(255,255,255))  # create white mask

        
        state = torch.get_rng_state()
        image = self.spatial_transform(image)
        torch.set_rng_state(state)
        mask = 1.0-self.spatial_transform(ImageOps.invert(mask))
        
        image = self.image_transform(image)

        return image, mask, label


def get_wbc_dataset_with_masks(type):
    
    mean = data['train']['mean']
    std = data['train']['std']
    
    assert type != 'val', 'no masks in validation set'

    spatial_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize((512,512), antialias=True)
    ])
    image_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
        
    
    data_path = data['train']['paths'][type]
    m = data['train']['paths'][type].split('/')
    m[-1] = 'mask'
    mask_path = os.path.join(*m)
    dataset = MaskDataset(data_root=data_path, mask_root=mask_path, spatial_transform=spatial_transform, image_transform=image_transform)
    return dataset

def apply_mask(images, masks, mean, std):

    noise = torch.randn_like(images)
    for c in range(images.shape[1]):
        noise[:, c] = noise[:, c] * std[c] + mean[c]

    masked_images = masks * images + (1 - masks) * noise

    return masked_images

if __name__ == "__main__":
    training_dataset = get_wbc_dataset_with_masks('wbc_100')
    mean = data['train']['mean']
    std = data['train']['std']
    # equal sampling from all classes
    targets_tensor = torch.tensor(training_dataset.targets)
    class_sample_count = torch.tensor([(targets_tensor == t).sum() for t in torch.unique(targets_tensor, sorted=True)])
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t] for t in training_dataset.targets])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(training_dataset, batch_size=1, sampler=sampler, num_workers=1)

    for step, (image_batch, mask_batch, labels_batch) in enumerate(train_loader):
        print(step)
        masked_img_batch = apply_mask(image_batch, mask_batch, mean, std)
        
        img = image_batch.cpu().squeeze()
        mask = mask_batch.cpu().squeeze()
        masked_img = masked_img_batch.cpu().squeeze()
        
        plt.figure()
        plt.subplot(131)
        plt.title('img')
        plt.imshow(denormalize_img(img, mean, std))
        plt.subplot(132)
        plt.title('mask')
        plt.imshow(torch.clamp(mask.permute((1,2,0)), 0, 1))
        plt.subplot(133)
        plt.title('masked img')
        plt.imshow(denormalize_img(masked_img, mean, std))
       
        plt.savefig(f'output/masking-{step}.png')
        
        if step > 15:
            break
        
    print('done')