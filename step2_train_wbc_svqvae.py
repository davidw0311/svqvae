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
            transforms.RandomRotation(180, fill=mean),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    
    path = data['train']['paths'][type]

    dataset = ImageFolder(root=path, transform=transform)
    return dataset
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training config")

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--alternating_epochs", type=int, required=True)
    args = parser.parse_args()
    
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    alternating_epochs = args.alternating_epochs
    save_every = args.save_every
    model_checkpoint = args.checkpoint
    dataset = args.dataset
    pretrain_dir = os.path.join(*model_checkpoint.split('/')[:-1])
    
    model_config = SimpleNamespace()
    with open(os.path.join(pretrain_dir, 'model_config.py'), 'r') as f:
        configs = f.read()
    exec(configs, vars(model_config))
    
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    directory_name = f"runs/train-svqvae-{dataset}-" + current_time
    os.makedirs(directory_name, exist_ok=True)
    checkpoints_directory = os.path.join(directory_name, "checkpoints")
    os.makedirs(checkpoints_directory, exist_ok=True)
    current_script_name = os.path.basename(__file__)
    shutil.copy2(current_script_name, directory_name)

    log_file = os.path.join(directory_name, f"run_{current_time}.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])
    logging.info(f"{args}")
    
    num_gpus = 1
    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        num_gpus = torch.cuda.device_count()
        logging.info(f"{num_gpus} gpus available")
        logging.info(f'using device {device_name} {device}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info(f'using device {device}')
    else:
        device = 'cpu'  
        logging.info(f'using device {device}')
    
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
    
    logging.info(f'checkpoint: {model_checkpoint}')
    logging.info(f'dataset: {dataset}')
    logging.info(f'input: {in_channel} x {img_size} x {img_size}')
    logging.info(f'num classes: {num_classes}')
    logging.info(f'# vaes: {num_vaes}')
    logging.info(f'vae channels: {vae_channels}')
    logging.info(f'vae res blocks: {res_blocks}')
    logging.info(f'vae res channels: {res_channels}')
    logging.info(f'vae embedding dims: {embedding_dims}')
    logging.info(f'codebook sizes: {codebook_size}')
    logging.info(f'decays: {decays}')
    
    checkpoint = torch.load(model_checkpoint, map_location='cpu')
    
    pretrain_weights = checkpoint['model_state_dict']
    
    # model_state = model.state_dict()
    # filtered_weights = {k: v for k, v in pretrain_weights.items() if k in model_state and v.shape == model_state[k].shape}
    # model_state.update(filtered_weights)
    # model.load_state_dict(model_state)
    
    model.load_state_dict(pretrain_weights)
    model = model.to(device)
    
    logging.info(f"current memory allocation: {torch.cuda.memory_allocated(device) / (1024 ** 3)}")
    logging.info(f'starting training from checkpoint {model_checkpoint}')
    
    logging.info(f'total number of parameters: { sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logging.info(f'training with batch size {batch_size} for {num_epochs} epochs')
    logging.info(f'alterating every {alternating_epochs} epochs')
    logging.info(f'saving checkpoint every {save_every} epochs')
    
    training_dataset = get_wbc_dataset(dataset)
    
    # equal sampling from all classes

    targets_tensor = torch.tensor(training_dataset.targets)
    class_sample_count = torch.tensor([(targets_tensor == t).sum() for t in torch.unique(targets_tensor, sorted=True)])
    weight = 1. / class_sample_count
    samples_weight = torch.tensor([weight[t] for t in training_dataset.targets])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(training_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
  

    testing_dataset = get_wbc_dataset('val')
    test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    

    # for finetuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    losses = {}

    losses[f'epoch_recon_loss'] = []
    losses[f'epoch_latent_loss'] = []
    losses[f'epoch_cls_loss'] = []
    losses[f'epoch_train_acc'] = []
    losses[f'epoch_test_acc'] = []

    latent_loss_weight = 0.25
    best_test_acc = 0
    
    show_every_dict = {
        'wbc_1': 1,
        'wbc_10': 10,
        'wbc_50': 50,
        'wbc_100': 50
    }
    
    show_every = show_every_dict[dataset]
    
    for e in range(1, num_epochs+1):
        logging.info( f"{f'starting epoch {e}':-^{50}}" )
        
        model.train()
        
        for k in losses:
            losses[k].append(0.0)
        
        mse_loss = torch.nn.MSELoss()
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        train_total = 0 
        for step, (image_batch, labels_batch) in enumerate(train_loader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            
            if e %alternating_epochs == 0:
                qt, qb, qj, diff, idt, idb = model.encode(image_batch, 0, num_vaes-1) 
                recon = model.decode(qj, num_vaes-1, -1)
                
                latent_loss = diff.mean()
                recon_loss = mse_loss(image_batch, recon)
                total_loss = latent_loss_weight*latent_loss + recon_loss
                total_loss.backward()
            else:
                recon_loss = torch.tensor(0)
                latent_loss = torch.tensor(0)
                
            preds = model.predict(image_batch)
            prediction_loss = cross_entropy_loss(preds, labels_batch)
            prediction_loss.backward()
     
            optimizer.step()
            
            losses['epoch_recon_loss'][-1] += recon_loss.item()
            losses['epoch_latent_loss'][-1] += latent_loss.item()
            losses['epoch_cls_loss'][-1] += prediction_loss.item()
            
            train_predictions = torch.argmax(preds, dim=1)
            losses['epoch_train_acc'][-1] += torch.sum(train_predictions == labels_batch).item()
            train_total += labels_batch.shape[0]
                
            if (step+1)%show_every == 0 :
                logging.info(f"avg cls loss after step {step+1}: {losses['epoch_cls_loss'][-1]/(step+1)}")
                logging.info(f"avg recon loss after step {step+1}: {losses['epoch_recon_loss'][-1]/(step+1)}")
                logging.info(f"avg latent loss after step {step+1}: {losses['epoch_latent_loss'][-1]/(step+1)}")
                logging.info(f"training acc after step {step+1}: {losses['epoch_train_acc'][-1]/(train_total)}")    
  
        # scheduler.step()

        
        losses['epoch_cls_loss'][-1] /= (step+1)
        losses['epoch_recon_loss'][-1] /= (step+1)
        losses['epoch_latent_loss'][-1] /= (step+1)

        losses['epoch_train_acc'][-1] /= train_total
        
        logging.info(f"epoch {e} training acc: {losses['epoch_train_acc'][-1]}")   
        
        ### testing metrics
        model.eval()
       
        test_total = 0 
        for step, (image_batch, labels_batch) in enumerate(test_loader):
            image_batch = image_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            preds = model.predict(image_batch)

            test_predictions = torch.argmax(preds, dim=1)
            
            losses['epoch_test_acc'][-1] += torch.sum(test_predictions == labels_batch).item()
            test_total += labels_batch.shape[0]

        
        losses['epoch_test_acc'][-1] /= test_total
        logging.info(f"epoch {e} testing acc: {losses['epoch_test_acc'][-1]}")
        
        if losses['epoch_test_acc'][-1] > best_test_acc:
            best_test_acc = losses['epoch_test_acc'][-1]
            
            if best_test_acc > 0.85:
                torch.save({
                    'model_state_dict':  model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': e
                }, f'{checkpoints_directory}/svqvae_best_{e}.pt')
            
        ### saving checkpoints
        if e%save_every == 0 or e == 1:
            torch.save({
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'epoch': e
            }, f'{checkpoints_directory}/svqvae_model_{e}.pt')
            
            with open(f'{checkpoints_directory}/losses.json', 'w') as f:
                json.dump(losses, f)
                
            logging.info(f'saving checkpoint and losses after episode {e}')
