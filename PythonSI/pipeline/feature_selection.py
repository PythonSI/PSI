from .base import Pipeline
from ..node import Data
import numpy as np
import numpy.typing as npt
from ..util import compute_p_value
from ..util.line_search import line_search
from typing import List
from ..test_statistics import (
    SFS_DATestStatistic,
    FSTestStatistic                               
)

class FeatureSelectionPipeline(Pipeline):
    def __init__(self, inputs: List[Data], output: Data, test_statistic: any):
        self.input_nodes = inputs
        self.output_node = output 
        self.test_statistic = test_statistic

    def __call__(
        self, 
        inputs: List[npt.NDArray[np.floating]],
        sigmas: List[npt.NDArray[np.floating]],
        verbose: bool = False
    ) -> npt.NDArray[np.floating]:
        """
        Run the feature selection pipeline.
        """
        for input_data, input_node in zip(inputs, self.input_nodes):
            input_node.update(input_data)

        selected_feature = self.output_node()
        if verbose:
            print(f"Selected feature: {selected_feature}")
        list_p_value = []
        for feature_id, _ in enumerate(selected_feature):
            if verbose:
                print(f"Testing feature {feature_id}")
            p_value = self.inference(feature_id=feature_id, selected_feature=selected_feature, sigmas=sigmas)
            
            if verbose:
                print(f"Feature {feature_id}: p-value = {p_value}")
            list_p_value.append(p_value)
        return selected_feature, list_p_value
            
    def inference(
        self, 
        feature_id: int, 
        sigmas: List[npt.NDArray[np.floating]],
        selected_feature: npt.NDArray[np.floating],
        ) -> float:
        """
        Run inference on a single feature.
        """

        test_statistic_direction, a, b, test_statistic, variance, deviation = self.test_statistic(selected_feature, feature_id, sigmas)
        
        # # For debug:
        # print(f"Test statistic: {test_statistic}")
        
        list_intervals, list_outputs = line_search(self, self.output_node, z_min=min(-20 * deviation, test_statistic), z_max=max(20 * deviation, test_statistic), step_size=1e-4)
        p_value = compute_p_value(test_statistic, variance, list_intervals, list_outputs, selected_feature)
        
        return p_value

