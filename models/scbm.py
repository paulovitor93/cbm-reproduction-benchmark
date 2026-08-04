import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, RelaxedBernoulli
    
class SCBM(nn.Module):
    """
    Stochastic Concept Bottleneck Model (SCBM).
    """

    def __init__(self, backbone, feature_dim, num_concepts, num_classes, mc_samples=50):
        super().__init__()
        self.backbone = backbone
        self.temperature = 1.0

        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.mc_samples = mc_samples

        # Mean of concept logits
        self.mu_head = nn.Linear(feature_dim, num_concepts)

        # Cholesky parameters
        self.cov_head = nn.Linear(feature_dim, num_concepts * (num_concepts + 1) // 2)

        # Target predictor
        self.classifier = nn.Linear(num_concepts, num_classes)

    def build_cholesky(self, vec):

        B = vec.size(0)
        C = self.num_concepts
        L = torch.zeros(B, C, C, device=vec.device, dtype=vec.dtype)

        idx = torch.tril_indices(C, C)

        L[:, idx[0], idx[1]] = vec

        diag_idx = torch.arange(C, device=vec.device)
        L[:, diag_idx, diag_idx] = (F.softplus(L[:, diag_idx, diag_idx]) + 1e-6)

        return L
    
    def sample_concepts(self, logits):
        dist = RelaxedBernoulli(temperature=logits.new_tensor(self.temperature), logits=logits)

        if self.training:
            soft = dist.rsample()
            hard = (soft > 0.5).float()
            concepts = hard.detach() - soft.detach() + soft
        else:
            concepts = (torch.sigmoid(logits) > 0.5).float()

        return concepts
        
    def sample_logits(self, mu, L):
        """
        Sample concept logits using the reparameterization trick.
        """

        c_dist = MultivariateNormal(loc=mu, scale_tril=L)
        eta_samples = c_dist.rsample((self.mc_samples,))
        eta_samples = eta_samples.permute(1, 2, 0)

        return eta_samples

    def concept_forward(self, features):
        """
        Predict concept distributions.
        """

        mu = self.mu_head(features)
        Lvec = self.cov_head(features)

        L = self.build_cholesky(Lvec)
        eta_samples = self.sample_logits(mu, L)

        concept_probs_samples = torch.sigmoid(eta_samples)
        concept_probs = concept_probs_samples.mean(dim=-1)
        
        
        Sigma = torch.bmm(L, L.transpose(1, 2))
        jitter = 1e-4 * torch.eye(self.num_concepts, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
        Sigma = Sigma + jitter

        return {
            "concept_probs": concept_probs,
            "concept_probs_samples": concept_probs_samples,
            "eta_samples": eta_samples,
            "Sigma": Sigma,
            "mu": mu,
            "L": L
        }

    def class_forward(self, eta_samples):
        """
        Predict task labels from sampled concepts.
        """

        mc_logits = []
        for m in range(self.mc_samples):
            sampled = self.sample_concepts(eta_samples[:, :, m])
            mc_logits.append(self.classifier(sampled))

        mc_logits = torch.stack(mc_logits, dim=-1)

        class_probs = torch.softmax(mc_logits, dim=1)
        class_probs = class_probs.mean(dim=-1)
        class_logits = torch.log(class_probs + 1e-8)

        return {
            "class_logits": class_logits,
            "class_probs": class_probs
        }

    def forward(self, x):
        features = self.backbone(x)
        concept_out = self.concept_forward(features)
        class_out = self.class_forward(concept_out["eta_samples"])

        return {
            "concept_probs": concept_out["concept_probs"],
            "concept_probs_samples": concept_out["concept_probs_samples"],
            "class_probs": class_out["class_probs"],
            "class_logits": class_out["class_logits"],
            "mu": concept_out["mu"],
            "L": concept_out["L"],
            "Sigma": concept_out["Sigma"],
            "eta_samples": concept_out["eta_samples"],
        }