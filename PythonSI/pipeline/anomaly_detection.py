from .base import Pipeline

class AnomalyDetectionPipeline(Pipeline):
    def __init__(self, *args):
        super().__init__(*args)

    def inference(self, X=None, X_source=None):
        pass
