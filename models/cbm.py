import torch
import torch.nn as nn

class CBM(nn.Module):
    """
    Standard Concept Bottleneck Model.
    """

    def __init__(self, backbone, num_concepts, num_classes,):
        super().__init__()

        self.backbone = backbone
        self.concept_head = nn.Linear(backbone.feature_dim, num_concepts)
        self.task_head = nn.Linear(num_concepts, num_classes)

    def forward(self, images):

        features = self.backbone(images)
        concept_logits = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_logits)
        class_logits = self.task_head(concept_probs)

        return {
            "features": features,
            "concept_logits": concept_logits,
            "concept_probs": concept_probs,
            "class_logits": class_logits,
        }

    def intervene(self, concepts, concept_values, mask):

        # Replace predicted concepts with intervention values where mask is True.
        intervened_concepts = torch.where(mask, concept_values, concepts)
        intervened_class_logits = self.task_head(intervened_concepts)

        return {
            "concepts": intervened_concepts,
            "class_logits": intervened_class_logits,
        }