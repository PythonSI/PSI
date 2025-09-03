import numpy as np
from ..pipeline import Pipeline
from ..node import Data

def line_search(
    pipeline: Pipeline,
    output_node: Data,
    z_min: float,
    z_max: float,
    step_size: float = 1e-4
):
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
        
    for i in range(len(list_intervals)-1):
        assert(list_intervals[i][1] <= list_intervals[i+1][0], "Intervals are overlapping in line search")
        
    return list_intervals, list_outputs