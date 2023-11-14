import torch
from torchvision import transforms
from step1_analyze_data import PretrainingDataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import os, shutil, datetime
import logging
from data.stats import data
import numpy as np
import argparse
from models.svqvae import SVQVAE
import json

def get_pretrain_dataset(name):
    info = data['pretrain'][name]
    mean = info['mean']
    std = info['std']
    
    if name=='cam16':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
            transforms.Resize((512,512), antialias=True)
        ])
    elif name=='prcc':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Normalize(mean, std),
            transforms.RandomCrop((512,512))
        ])
    
    datasets = []
    for folder in info['paths']:
        dataset = PretrainingDataset(img_dir=folder, transform=transform)
        datasets.append(dataset)
        
    dataset = ConcatDataset(datasets)
    return dataset
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training config")

    parser.add_argument("--batch_size", type=int, help="batch_size", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save_every", type=int, required=True)
    parser.add_argument("--level_stagger_epochs", type=int, required=True)
    parser.add_argument("--method", type=str,choices=["recon_level", "recon_all"], required=True)
    parser.add_argument("--description", type=str, required=True)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_epochs = args.epochs
    save_every = args.save_every
    level_stagger_epochs = args.level_stagger_epochs
    method = args.method
    
    current_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
    directory_name = f"runs/pretrain-svqvae-{method}-b{batch_size}-e{num_epochs}-s{level_stagger_epochs}-" + current_time
    os.makedirs(directory_name, exist_ok=True)
    checkpoints_directory = os.path.join(directory_name, "checkpoints")
    os.makedirs(checkpoints_directory, exist_ok=True)
    current_script_name = os.path.basename(__file__)
    shutil.copy2(current_script_name, directory_name)
    model_file1 = 'models/svqvae.py'
    model_file2 = 'models/vqvae.py'
    shutil.copy2(model_file1, directory_name)
    shutil.copy2(model_file2, directory_name)
    log_file = os.path.join(directory_name, f"run_{current_time}.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])

    if torch.cuda.is_available():
        device = torch.device('cuda:0') 
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        logging.info(f'using device {device_name} {device}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info(f'using device{device}')
    else:
        device = 'cpu'  
        logging.info(f'using device {device}')
    
    
    img_size=512
    in_channel=3
    num_classes=5
    num_vaes=3
    vae_channels=[128, 128, 128]
    res_blocks=[4,4,4]
    res_channels=[64,64,64]
    embedding_dims=[3,3,3]
    codebook_size=[512,512,512]
    decays=[0.99,0.99,0.99]
    
    with open(f'{checkpoints_directory}/model_config.py', 'w') as f:
        f.write(f'img_size={img_size}\n')
        f.write(f'in_channel={in_channel}\n')
        f.write(f'num_classes={num_classes}\n')
        f.write(f'num_vaes={num_vaes}\n')
        f.write(f'vae_channels={vae_channels}\n')
        f.write(f'res_blocks={res_blocks}\n')
        f.write(f'res_channels={res_channels}\n')
        f.write(f'embedding_dims={embedding_dims}\n')
        f.write(f'codebook_size={codebook_size}\n')
        f.write(f'decays={decays}\n')
        
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
    
    logging.info(args.description)
    logging.info(f'method: {method}')
    logging.info(f'input: {in_channel} x {img_size} x {img_size}')
    logging.info(f'num classes: {num_classes}')
    logging.info(f'# vaes: {num_vaes}')
    logging.info(f'level staggering epochs: {level_stagger_epochs}')
    logging.info(f'vae channels: {vae_channels}')
    logging.info(f'vae res blocks: {res_blocks}')
    logging.info(f'vae res channels: {res_channels}')
    logging.info(f'vae embedding dims: {embedding_dims}')
    logging.info(f'codebook sizes: {codebook_size}')
    logging.info(f'decays: {decays}')
    
    model = model.to(device)
    
    logging.info(f'total number of parameters: { sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logging.info(f'training with batch size {batch_size} for {num_epochs} epochs')
    logging.info(f'saving checkpoint every {save_every} epochs')
    
    cam16_ds =get_pretrain_dataset('cam16')
    prcc_ds = get_pretrain_dataset('prcc')
    
    merged_ds = ConcatDataset([cam16_ds, prcc_ds])
    weights = [1.0]*len(cam16_ds) + [4.0]*len(prcc_ds)
    sampler = WeightedRandomSampler(weights, num_samples = len(merged_ds), replacement=True)
    
    pretrain_loader = DataLoader(merged_ds, batch_size=batch_size, sampler=sampler, num_workers=2)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    losses = {}
    for l in range(num_vaes):
        losses[f'epoch_recon_loss_vae{l}'] = []
        losses[f'epoch_latent_loss_vae{l}'] = []
        
    latent_loss_weight = 0.25
    
    mse_loss = torch.nn.MSELoss()
    for e in range(1, num_epochs+1):
        for k in losses:
            losses[k].append(0.0)
        
        if level_stagger_epochs == 0 :
            training_level = num_vaes-1
        else:
            training_level = min((e-1)//level_stagger_epochs, num_vaes-1)
        
        logging.info( f"{f'starting epoch {e}, training to vae level {training_level}':-^{100}}" )
        epoch_loss = 0 
        for step, image_batch in enumerate(pretrain_loader):
            image_batch = image_batch.to(device)
            
            optimizer.zero_grad()
            
            for level in range(training_level+1): 
                
                if method == 'recon_level':
                    if level == 0 :
                        input = image_batch
                    else:
                        _, _, input, _, _, _ = model.encode(image_batch, 0, level-1) # encode to previous level
                    
                    qt, qb, qj, diff, idt, idb = model.encode(input, level, level) 
                    recon = model.decode(qj, level, level-1)
                    
                elif method == 'recon_all':
                    input = image_batch
                    qt, qb, qj, diff, idt, idb = model.encode(input, 0, level) 
                    recon = model.decode(qj, level, -1)
                    
                latent_loss = diff.mean()
                recon_loss = mse_loss(input, recon)
                
                total_loss = latent_loss_weight*latent_loss + recon_loss
                
                total_loss.backward()
                
                losses[f'epoch_recon_loss_vae{level}'][-1] += recon_loss.item()
                losses[f'epoch_latent_loss_vae{level}'][-1] += latent_loss.item()
            
                
            optimizer.step()
            
            if (step+1)%10 == 0 :
                logging.info(f"{f'step {step+1}':-^{50}}")
                for k, v in losses.items():
                    logging.info(f'avg {k}: {v[-1]/(step+1)}')
  
        scheduler.step()
        
        for k in losses:
            losses[k][-1]/=(step+1)
        
        if e%save_every == 0 or e == 1:
            torch.save({
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': e
            }, f'{checkpoints_directory}/model_{e}.pt')
            
            with open(f'{checkpoints_directory}/losses.json', 'w') as f:
                json.dump(losses, f)
                
            logging.info(f'saving checkpoint and losses after episode {e}')