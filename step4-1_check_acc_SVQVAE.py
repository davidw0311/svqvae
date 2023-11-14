import torch
from models.svqvae import SVQVAE
from torchvision import transforms
from step1_analyze_data import PretrainingDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from data.stats import data
from data.constants import *
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from types import SimpleNamespace
import os

def denormalize_img(img, mean, std):
    for t,m,s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    img = torch.clamp(img, 0,1)
    return img.permute((1,2,0))

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
    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'  
    print('using device ', device)
    
    model_checkpoint = 'checkpoints/train-svqvae-wbc_100-1024_023002/checkpoints/svqvae_best_95.pt'
    # model_checkpoint = 'checkpoints/train-svqvae-wbc_50-1024_022839/checkpoints/svqvae_best_92.pt'
    # model_checkpoint = 'checkpoints/train-svqvae-wbc_10-1024_022301/checkpoints/svqvae_best_95.pt'
    # model_checkpoint = 'checkpoints/train-svqvae-wbc_1-1024_012902/checkpoints/svqvae_best_98.pt'
    
    plt.rcParams['font.size'] = 14
    ds = ''
    if 'wbc_100' in model_checkpoint:
        ds = 'WBC 100'
    elif 'wbc_50' in model_checkpoint:
        ds = 'WBC 50'
    elif 'wbc_10' in model_checkpoint:
        ds = 'WBC 10'
    elif 'wbc_1' in model_checkpoint:
        ds = 'WBC 1'

    pretrain_dir = os.path.join(*model_checkpoint.split('/')[:-1])
    
    model_config = SimpleNamespace()
    with open(os.path.join(pretrain_dir, 'model_config.py'), 'r') as f:
        configs = f.read()
    exec(configs, vars(model_config))
    
    img_size=model_config.img_size
    in_channel=model_config.in_channel
    num_classes=model_config.num_classes
    num_vaes=model_config.num_vaes
    vae_channels=model_config.vae_channels
    res_blocks=model_config.res_blocks
    res_channels=model_config.res_channels
    embedding_dims=model_config.embedding_dims
    codebook_size=model_config.codebook_size
    decays=model_config.decays
    
    model = SVQVAE(
        img_size=img_size,
        in_channel=in_channel,
        num_classes=num_classes,
        num_vaes=num_vaes,
        vae_channels=vae_channels,
        res_blocks=res_blocks,
        res_channels=res_channels,
        embedding_dims=embedding_dims,
        codebook_size=codebook_size,
        decays=decays
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    pretrain_weights = checkpoint['model_state_dict']
    model.load_state_dict(pretrain_weights)

    model = model.to(device)
    
    with open(os.path.join(pretrain_dir, 'losses.json'), 'r') as f:
        losses = json.load(f)
    
    
    for name in losses:
        plt.clf()
        loss = []
        epochs = []
        for idx, l in enumerate(losses[name]):
            if l != 0:
                loss.append(l)
                epochs.append(idx+1)
        plt.plot(epochs, loss, '-o', label=name)
        plt.xlim(0, len(losses[name])+1)
        if 'recon' in name or 'latent' in name:
            plt.ylim(0,1)
        if 'acc' in name:
            print(ds, name, 'best = ', max(loss))
            plt.ylabel('Accuracy')
        else:
            plt.ylabel('Loss')
        plt.title(f'Finetune {ds}: {name}')
        plt.xlabel('Epoch')
        
        plt.savefig(f"output/finetune-{ds.replace(' ','-')}-{name}.png") 


    mean = [0.7048, 0.5392, 0.5885]
    std = [0.1626, 0.1902, 0.0974]
    
    testing_dataset = get_wbc_dataset('val')
    test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=True, num_workers=2)

    model.eval()
    y_pred = []
    y_actual = []
    correct = 0 
    total = 0 
    for step, (image_batch, labels_batch) in tqdm(enumerate(test_loader)):
        
        image_batch = image_batch.to(device)
        labels_batch = labels_batch.to(device)
        
        preds = model.predict(image_batch)
        
        test_prediction = torch.argmax(preds, dim=1)
        y_pred.append(test_prediction.item())
        y_actual.append(labels_batch.item())
        
        correct += torch.sum(test_prediction == labels_batch).item()
        total += 1
            
        pred_label = CLASSES[test_prediction.item()]
        actual_label = CLASSES[labels_batch.item()]
            
                
    cm = confusion_matrix(y_actual, y_pred)
    df_cm = pd.DataFrame(cm, index = [c for c in CLASSES],
                    columns = [c for c in CLASSES])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, cbar=False)
    plt.title(f'Confusion Matrix for WBC Testing dataset, trained on {ds}')
    plt.ylabel('Ground Truth Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.savefig(f"output/finetune-{ds.replace(' ', '-')}-confusion-matrix.png")

        
    
    