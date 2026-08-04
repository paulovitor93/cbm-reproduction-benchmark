import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pem import PEM
    
class ProbCBM(nn.Module):
  """
  Probabilistic Concept Bottleneck Model (ProbCBM).
  
  This implementation follows the probabilistic embedding formulation
  proposed in the original paper, using Monte Carlo sampling and
  anchor-based concept and class representations.
  """
  def __init__(self, backbone, feature_dim, num_concepts, num_classes, concept_dim=16, class_dim=128, mc_samples=50):
    super().__init__()

    self.backbone = backbone
    self.num_concepts = num_concepts        
    self.num_classes = num_classes
    
    # Dimension of each concept embedding
    self.concept_dim = concept_dim
    # Dimension of class embedding
    self.class_dim = class_dim
    # Number of Monte-Carlo samples
    self.mc_samples = mc_samples

    # PEMs for each concept
    self.pems = nn.ModuleList([PEM(feature_dim, concept_dim) for _ in range(num_concepts)])

    # Concept anchors
    self.pos_anchor = nn.Parameter(torch.randn(num_concepts, concept_dim))
    nn.init.normal_(self.pos_anchor, mean=0.0, std=1.0 / (concept_dim ** 0.5))
    
    self.neg_anchor = nn.Parameter(torch.randn(num_concepts, concept_dim))
    nn.init.normal_(self.neg_anchor, mean=0.0, std=1.0 / (concept_dim ** 0.5))

    # Learnable scaling parameter for concept probabilities.
    self.alpha = nn.Parameter(torch.tensor(5.0))

    # Projection from concatenated concept embeddings to class embedding space
    self.project = nn.Linear(num_concepts * concept_dim, class_dim)

    # class anchors
    self.class_anchor = nn.Parameter(torch.randn(num_classes, class_dim))
    nn.init.normal_(self.class_anchor, mean=0.0, std=1.0 / (class_dim ** 0.5))

    # Scaling parameter from Eq.(5)
    # how confident the class probability becomes.
    self.beta = nn.Parameter(torch.tensor(10.0))
  
  # Reparametrization
  def sample(self, mu, log_sigma):
    sigma = torch.exp(log_sigma)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps
  
  # Eq.(3): Concept probability
  def concept_probability(self, z, idx):
    pos = self.pos_anchor[idx]
    neg = self.neg_anchor[idx]

    # Root mean square distance.
    d_pos = torch.sqrt(((z - pos) ** 2).mean(-1) + 1e-10)
    d_neg = torch.sqrt(((z - neg) ** 2).mean(-1) + 1e-10)

    distance_difference = d_neg - d_pos
    logits = self.alpha * distance_difference

    return torch.sigmoid(logits)
  
  # Concept predictor
  def concept_forward(self, features):
    concept_probs = []
    mus = []
    log_sigmas = []
    sampled_z = []
    # Process each concept independently
    for i in range(self.num_concepts):
        # Predict Gaussian parameters
        mu, log_sigma = self.pems[i](features)

        mus.append(mu)
        log_sigmas.append(log_sigma)

        prob_sum = 0
        samples = []
        
        # Monte-Carlo estimation
        for _ in range(self.mc_samples):
            # Sample concept embedding
            z = self.sample(mu, log_sigma)
            samples.append(z)

            # Compute concept probability
            prob_sum += self.concept_probability(z, i)
        
        # Average MC probabilities
        concept_probs.append(prob_sum / self.mc_samples)
        sampled_z.append(samples)
    
    # (B, num_concepts, mc_samples, concept_dim)
    sampled_z = torch.stack([torch.stack(samples, dim=1) for samples in sampled_z], dim=1)

    return {
        "concept_probs": torch.stack(concept_probs, dim=1),
        "mu": torch.stack(mus, dim=1),
        "log_sigma": torch.stack(log_sigmas, dim=1),
        "sampled_z": sampled_z
    }
  
  def concept_uncertainty(self, log_sigma):
    """
    Estimate concept uncertainty from the predicted log-variance.
    """

    return torch.exp(log_sigma.mean(dim=-1))
  
  # Class predictor
  def class_forward(self, sampled_z,  concept_labels=None, p_replace=0.5):
    prob_sum = 0
    logits_sum = 0
    last_embedding = None
    
    # Monte-Carlo integration
    for s in range(self.mc_samples):
        embeddings = []

        # get the s-th sample for every concept
        for c in range(self.num_concepts):
          z = sampled_z[:, c, s, :]
          
          # Anchor replacement trick (Algorithm 1)
          if self.training and concept_labels is not None:

              # Independent replacement mask for each image
              mask = (torch.rand(concept_labels.shape[0], 1, device=concept_labels.device,) < p_replace).float()

              positive = self.pos_anchor[c]
              negative = self.neg_anchor[c]

              gt = concept_labels[:, c].float().unsqueeze(1)

              # Ground-truth anchor embedding
              anchor = gt * positive + (1 - gt) * negative

              # Replace sampled embedding with probability p_replace
              z = mask * anchor + (1 - mask) * z

          embeddings.append(z)

        # concatenate concept embeddings
        concat = torch.cat(embeddings, dim=1)

        # project to class embedding
        h = self.project(concat)
        h = F.normalize(h, p=2, dim=-1)
        last_embedding = h

        # distance to class anchors
        diff = (h.unsqueeze(1) - self.class_anchor.unsqueeze(0))
        dist = torch.sqrt((diff ** 2).mean(-1) + 1e-10)
        logits = -self.beta * dist

        # Eq.(5): compute probabilities
        probs = torch.softmax(logits, dim=1)
        # save last Monte-Carlo logits
        logits_sum += logits
        prob_sum += probs
    
    # Monte-Carlo average
    class_probs = prob_sum / self.mc_samples
    class_logits = logits_sum / self.mc_samples

    return {
        "class_probs": class_probs,
        "class_logits": class_logits,
        "class_embedding": last_embedding,
    }

  def forward(self, x, concept_labels=None, p_replace=0.5):
    features = self.backbone(x)
    concept_out = self.concept_forward(features)
    class_out = self.class_forward(concept_out["sampled_z"], concept_labels=concept_labels, p_replace=p_replace)

    return {
        "concept_probs": concept_out["concept_probs"],
        "class_probs": class_out["class_probs"],
        "class_logits": class_out["class_logits"],
        "mu": concept_out["mu"],
        "log_sigma": concept_out["log_sigma"],
        "concept_uncertainty": self.concept_uncertainty(concept_out["log_sigma"]),
        "class_embedding": class_out["class_embedding"]
        }