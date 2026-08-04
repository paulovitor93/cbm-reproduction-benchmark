import torch
import torch.nn as nn
import torch.nn.functional as F

class PEM(nn.Module):
    """
    Probabilistic Embedding Module used by ProbCBM.
    """
    def __init__(self,input_dim,embedding_dim=16):
        super().__init__()

        self.mean_head = nn.Linear(input_dim, embedding_dim)
        self.logsigma_head = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        mu = self.mean_head(x)
        
        # Normalize embedding means to lie on the unit hypersphere.
        mu = F.normalize(mu, p=2, dim=-1)
        
        log_sigma = self.logsigma_head(x)
        log_sigma = torch.clamp(log_sigma, -10, 10)

        return mu, log_sigma
    
def kl_loss(mu, log_sigma):
    sigma = torch.exp(log_sigma)
    kl = 0.5 * (sigma.pow(2) + mu.pow(2) - 1 - 2 * log_sigma)
    return kl.mean()