import os
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.optim.lr_scheduler as lrs
from pytorch_lightning import LightningModule

from model import UNet
from diffusion import Diffusion

class MInterface(LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.image_size = args.image_size
        self.batch_size = args.batch_size
        
        self.lr = args.lr
        self.lr_scheduler = args.lr_scheduler
        self.lr_decay_steps = args.lr_decay_steps
        self.lr_decay_rate = args.lr_decay_rate
        self.lr_decay_min_lr = args.lr_decay_min_lr
        
        self.model = UNet(1000, 3, 3)
        self.diffu = Diffusion(1000, args.image_size, self.model, "cuda:0")
    
    def forward(self, img):
        return self.model(img)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        if self.lr_scheduler is None:
                return optimizer
        else:
            if self.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer, 
                                       step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)
            elif self.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer, 
                                                  T_max=self.lr_decay_steps, eta_min=self.lr_decay_min_lr)
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        pokemon, item_image = batch
        z_bar = torch.randn_like(item_image)
        loss = self.diffu.loss(item_image, z_bar)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_T = torch.randn(self.batch_size, 3, self.image_size, self.image_size).to(self.device)
        z_bar = torch.randn_like(x_T)
        x_0 = self.diffu.sample(x_T, z_bar)
        
        return x_0
        
    def test_step(self, batch, batch_idx):
        
        return self.validation_step(batch, batch_idx)
    
    def predict_step(self, batch, batch_idx):
        
        return self.validation_step(batch, batch_idx)
        