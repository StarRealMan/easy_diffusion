import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Diffusion():
    def __init__(self, T, image_size, model, device):
        super().__init__()
        self.T = T
        self.device = device
        self.image_size = image_size
        
        alpha = 1 - 0.02 * torch.arange(1, T+1, device = self.device) / T
        beta = 1 - alpha
        alpha_bar = torch.cumprod(alpha, dim = 0)
        alpha_bar_prev = F.pad(alpha_bar, [1, 0], value=1)[:T]
        
        self.loss_coe_0 = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)
        self.loss_coe_1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1)
        
        self.p_sample_coe_0 = torch.sqrt(1 / alpha_bar).view(-1, 1, 1, 1)
        self.p_sample_coe_1 = torch.sqrt(1 / alpha_bar - 1).view(-1, 1, 1, 1)
        
        self.q_sample_coe_0 = (torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)).view(-1, 1, 1, 1)
        self.q_sample_coe_1 = (torch.sqrt(alpha) * (1. - alpha_bar_prev) / (1. - alpha_bar)).view(-1, 1, 1, 1)
        
        self.q_sample_var = (beta * (1. - alpha_bar_prev) / (1. - alpha_bar)).view(-1, 1, 1, 1)
        self.q_sample_var = beta.view(-1, 1, 1, 1)

        self.model = model
        self.objective = nn.MSELoss(reduction = 'sum')
        
    def loss(self, x_0, z_bar):
        batch_size = x_0.shape[0]
        t = torch.randint(1, self.T + 1, size=(batch_size, ), device=self.device)
        
        x_t = self.loss_coe_0[t-1] * x_0 + self.loss_coe_1[t-1] * z_bar
        
        loss = self.objective(self.model(x_t, t-1), z_bar)
        
        return loss
        
    def sample(self, x_T, z_bar):
        batch_size = x_T.shape[0]
        x_t = x_T
        for t in tqdm(range(self.T, 0, -1)):
            t = torch.tensor([t] * batch_size).to(self.device)
            
            x_0_hat = self.p_sample_coe_0[t-1] * x_t - self.p_sample_coe_1[t-1] * self.model(x_t, t-1)
            
            # print(self.p_sample_coe_0[t-1][0])
            # print(self.model(x_t, t-1).mean())
            
            # print(self.p_sample_coe_1[t-1][0])
            # print(x_t.mean())
            
            # print(x_0_hat.mean())
            
            if t[0] == 1:
                z_bar = torch.zeros_like(z_bar)
                
            x_t = self.q_sample_coe_0[t-1] * x_0_hat + self.q_sample_coe_1[t-1] * x_t + self.q_sample_var[t-1] * z_bar

        x_0 = x_t
        
        return x_0