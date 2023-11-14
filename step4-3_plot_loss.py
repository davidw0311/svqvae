import numpy as np
import matplotlib.pyplot as plt
import json

checkpoint_file = 'checkpoints/losses.json'
with open(checkpoint_file,'r') as f:
    losses = json.load(f)
    

for name in losses:
    plt.clf()
    plt.plot(losses[name], '-o')
    plt.title(name)
    plt.savefig(f"output/{name}.png") 
    
# train_epoch_losses = losses['train_']
# step_losses = np.load('runs/1016_105521/checkpoints/step_loss.npy')


# plt.plot(step_losses, '-o')
# plt.title('step losses for 1016_105521')
# plt.xlabel('step')
# plt.ylabel('loss')
# plt.savefig('output/step_losses.png')

# plt.clf()
# plt.plot(epoch_losses, '-o')
# plt.title('epoch losses for 1016_105521')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('output/epoch_losses.png')