import torch
from torch import nn
from torch.nn import functional as F
from models.vqvae import VQVAE






class SVQVAE(nn.Module):
    def __init__(
        self,
        img_size,
        in_channel=3,
        num_classes=5,
        num_vaes=3,
        vae_channels=[128, 128, 128],
        res_blocks=[2,2,2],
        res_channels=[32,32,32],
        embedding_dims=[1,1,1],
        codebook_size=[512,512,512],
        decays=[0.99, 0.99, 0.99]
    ):
        super().__init__()
        assert img_size/(4**num_vaes) > 1, "too many vaes, compression to less than 1 pixel"
        self.num_vaes = num_vaes
        self.models = nn.ModuleList([])
        in_channels = [in_channel] + [i*2 for i in embedding_dims[:-1]]
        for i in range(num_vaes):
            self.models.append(
                VQVAE(
                    in_channel=in_channels[i], 
                    channel=vae_channels[i], 
                    n_res_block=res_blocks[i], 
                    n_res_channel=res_channels[i], 
                    embed_dim=embedding_dims[i], 
                    n_embed=codebook_size[i], 
                    decay=decays[i]
                )
            )
        self.smallest_encoding = (img_size//(4**num_vaes))**2*embedding_dims[-1]*2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.smallest_encoding, self.smallest_encoding*2),
            nn.GELU(),
            nn.Linear(self.smallest_encoding*2, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, input, level):

        _, _, last_latent, _, _, _ = self.encode(input, level-1)
        
        quant_t, quant_b, diff, _, _ = self.models[level].encode(last_latent)
        recon_latent = self.models[level].decode(quant_t, quant_b)

        return last_latent, recon_latent, diff

    def encode(self, input, from_level, to_level):
        joint_quant = input
        for i in range(from_level, to_level+1):
            quant_t, quant_b, joint_quant, diff, id_t, id_b = self.models[i].encode(joint_quant)
        return quant_t, quant_b, joint_quant, diff, id_t, id_b
    
    
    def decode(self, joint_quant, from_level, to_level):
        for i in range(from_level, to_level, -1):
            joint_quant = self.models[i].decode(joint_quant)
            
        return joint_quant
            
    def predict(self, input):
        quant_t, quant_b, joint_quant, diff, id_t, id_b = self.encode(input, 0, self.num_vaes-1)
        probs = self.classifier(joint_quant)
        return probs
    