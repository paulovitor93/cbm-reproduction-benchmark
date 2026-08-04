class SceneValidator:
    """
    Validates generated scenes using the concept extractor and rule-based classifier.
    """
    def __init__(self, extractor, classifier):
        self.extractor = extractor
        self.classifier = classifier

    def predicted_class(self, scene):
        concepts = self.extractor.extract(scene)
        return self.classifier.predict(concepts)
    
    def validate(self, scene, expected_class):
        predicted = self.predicted_class(scene)
        
        return predicted == expected_class, predicted