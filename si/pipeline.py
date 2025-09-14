from si.node import Data
import numpy as np
import numpy.typing as npt
from si.util import compute_p_value
from typing import List


def line_search(output_node: Data, z_min: float, z_max: float, step_size: float = 1e-4):
    list_intervals = []
    list_outputs = []
    z = z_min
    while z < z_max:
        output, _, _, interval_of_z = output_node.inference(z=z)

        interval_of_z = [max(interval_of_z[0], z_min), min(interval_of_z[1], z_max)]

        list_intervals.append(interval_of_z)
        list_outputs.append(output)

        # # For debug:
        # print(f"z: {z}, interval: {interval_of_z}, output: {output}")

        z = interval_of_z[1] + step_size

    for i in range(len(list_intervals) - 1):
        assert list_intervals[i][1] >= list_intervals[i + 1][0] + 1e-13, (
            f"Intervals are overlapping in line search: {list_intervals[i]} and {list_intervals[i + 1]}"
        )

    return list_intervals, list_outputs


class Pipeline:
    r"""Selective inference for Feature selection pipeline.

    Parameters
    ----------
    inputs : list of Data
        Input datas for the pipeline
    output : Data
        Output data for the pipeline
    test_statistic : object
        Chosen test statistic for inference

    Attributes
    ----------
    input_nodes : list of Data
        Input data nodes
    output_node : Data
        Output data node
    test_statistic : object
        Test statistic computation object
    """

    def __init__(self, inputs: List[Data], output: Data, test_statistic: any):
        self.input_nodes = inputs
        self.output_node = output
        self.test_statistic = test_statistic

    def __call__(
        self,
        inputs: List[npt.NDArray[np.floating]],
        covariances: List[npt.NDArray[np.floating]],
        verbose: bool = False,
    ) -> npt.NDArray[np.floating]:
        r"""Execute the pipeline and inference on the observed output.

        Parameters
        ----------
        inputs : list of array-like
            Input data arrays corresponding to input_nodes
        covariances : list of array-like
            Covariance matrices for noise in each input
        verbose : bool, optional
            Whether to print detailed progress information, default False

        Returns
        -------
        outputs : array-like, shape (k,)
            Output data
        p_values : list of float
            Selective p-values for the output data
        """
        for input_data, input_node in zip(inputs, self.input_nodes):
            input_node.update(input_data)

        output = self.output_node()
        if verbose:
            print(f"Selected output: {output}")
        list_p_value = []
        for output_id, _ in enumerate(output):
            if verbose:
                print(f"Testing feature {output_id}")
            p_value = self.inference(
                output_id=output_id, output=output, covariances=covariances
            )

            if verbose:
                print(f"Feature {output_id}: p-value = {p_value}")
            list_p_value.append(p_value)
        return output, list_p_value

    def inference(
        self,
        output_id: int,
        covariances: List[npt.NDArray[np.floating]],
        output: npt.NDArray[np.floating],
    ) -> float:
        r"""Perform selective inference for a single specific data based on
        the chosen hypothesis and test statistic.

        For example, testing if a specific selected feature has a non-zero coefficient

        Parameters
        ----------
        feature_id : int
            Index of the feature to test (within selected features)
        covariances : list of array-like
            Covariance matrices for noise in each input
        output : array-like, shape (k,)
            Array of selected feature indices

        Returns
        -------
        p_value : float
            Selective p-value for the data
        """

        test_statistic_direction, a, b, test_statistic, variance, deviation = (
            self.test_statistic(output, output_id, covariances)
        )

        # # For debug:
        # print(f"Test statistic: {test_statistic}")

        list_intervals, list_outputs = line_search(
            self.output_node,
            z_min=min(-20 * deviation, test_statistic),
            z_max=max(20 * deviation, test_statistic),
            step_size=1e-4,
        )
        p_value = compute_p_value(
            test_statistic, variance, list_intervals, list_outputs, output
        )

        return p_value
