import torch
import torch.nn as nn

class CEM(nn.Module):
    """
    Concept Embedding Model (CEM).
    """

    def __init__(self, backbone, num_concepts, num_classes, embedding_size=16,):
        super().__init__()

        self.backbone = backbone
        self.num_concepts = num_concepts
        self.embedding_size = embedding_size
        
        hidden = backbone.feature_dim

        self.pos_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, embedding_size),
                nn.LeakyReLU()
            )
            for _ in range(num_concepts)
        ])

        self.neg_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, embedding_size),
                nn.LeakyReLU()
            )
            for _ in range(num_concepts)
        ])

        # Shared concept scorer.
        self.scorer = nn.Linear(2 * embedding_size, 1)
        
        self.classifier = nn.Linear(embedding_size * num_concepts, num_classes,)

    def forward(self, x):

        h = self.backbone(x)

        concept_logits = []
        concept_probs = []
        predicted_concept_embeddings = []

        for i in range(self.num_concepts):
            c_plus = self.pos_embeddings[i](h)
            c_minus = self.neg_embeddings[i](h)

            score_input = torch.cat([c_plus, c_minus], dim=1,)
            logit = self.scorer(score_input)
            prob = torch.sigmoid(logit)

            predicted_concept_embedding = (prob * c_plus + (1 - prob) * c_minus)

            concept_logits.append(logit)
            concept_probs.append(prob)
            predicted_concept_embeddings.append(predicted_concept_embedding)

        concept_logits = torch.cat(concept_logits, dim=1,)
        concept_probs = torch.cat(concept_probs, dim=1,)
        concept_bottleneck = torch.cat(predicted_concept_embeddings, dim=1,)
        class_logits = self.classifier(concept_bottleneck)

        return {
            "class_logits": class_logits,
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            "concept_bottleneck": concept_bottleneck,
        }