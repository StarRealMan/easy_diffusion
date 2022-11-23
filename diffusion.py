import torch
import torch.nn as nn
from tqdm import tqdm

class Diffusion():
    def __init__(self, T, image_size, model, device):
        super().__init__()
        self.T = T
        self.device = device
        self.image_size = image_size
        
        self.alpha = 1 - 0.02 * torch.arange(1, T+1, device = self.device) / T
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)
        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.sqrt_alpha_bar = torch.cumprod(self.sqrt_alpha, dim = 0)
        
        self.beta = 1 - self.alpha
        self.beta_bar = 1 - self.alpha_bar
        self.sqrt_beta_bar = torch.sqrt(self.beta_bar)
        
        self.sigma = self.beta / self.beta_bar * \
                     torch.cat([torch.tensor([0.0], device = self.device), self.beta_bar[:-1]])
        
        self.model = model
        self.objective = nn.MSELoss(reduction = 'sum')
        
    def loss(self, x_0, z_bar):
        batch_size = x_0.shape[0]
        t = torch.randint(1, self.T + 1, size=(batch_size, ), device=self.device)
        x_t = (self.sqrt_alpha_bar[t-1, None, None, None] * x_0 +
               self.sqrt_beta_bar[t-1, None, None, None] * z_bar)
        loss = self.objective(self.model(x_t, t-1), z_bar)
        
        return loss
        
    def sample(self, x_T, z_bar):
        x_t = x_T
        for t in tqdm(range(self.T, 0, -1)):
            t = torch.tensor([t]).to(self.device)
            
            x_0_hat = (x_t - self.sqrt_beta_bar[t-1, None, None, None] * self.model(x_t, t-1)) / \
                       self.sqrt_alpha_bar[t-1, None, None, None]
            x_t_1 = (self.sqrt_alpha[t-1, None, None, None] * self.beta_bar[t-1-1, None, None, None] * x_t + \
                    self.sqrt_alpha_bar[t-1-1, None, None, None] * x_0_hat) / self.beta_bar[t-1, None, None, None] + \
                    self.sigma[t-1, None, None, None] * z_bar
            x_t = x_t_1
        
        x_0 = x_t
        
        return x_0