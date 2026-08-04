from .OLDbackbone import ResNet18Backbone
from .OLDbase_cbm import BaseCBM


def build_model(
    num_classes: int,
    num_concepts: int,
    pretrained: bool = True,
    use_concept_bottleneck: bool = False,):
    
    backbone = ResNet18Backbone(
        pretrained=pretrained
    )

    model = BaseCBM(
        backbone=backbone,
        num_classes=num_classes,
        num_concepts=num_concepts,
        use_concept_bottleneck=use_concept_bottleneck,
    )

    return model