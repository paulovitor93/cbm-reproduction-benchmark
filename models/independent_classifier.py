import torch
import torch.nn as nn

class IndependentClassifier(nn.Module):
    """
    Linear classifier used by the Independent CBM.
    """

    def __init__(self, num_concepts: int = 112, num_classes: int = 200):
        super().__init__()

        self.num_concepts = num_concepts
        self.num_classes = num_classes

        self.task_head = nn.Linear(num_concepts, num_classes)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        return self.task_head(concepts)