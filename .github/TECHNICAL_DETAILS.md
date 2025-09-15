# Technical Detail

## Module structure

The module is organized around three main components: **Pipeline**, **Data**, and **Method**.

- **Pipeline**  
  The pipeline is represented as a directed graph, where **Data** and **Method** are the nodes and their relationships form the edges.  
  This graph structure captures the sequential nature of selective inference: each step depends on the successful completion of the previous step. The recursive design allows the process to be modeled and managed systematically.

- **Data node**  
  - Stores the observed data.  
  - Contains a reference to its parent **Method node**, indicating the computation that produced it.  
  - May include parameters *(a, b)* to determine whether the data is parameterized.  

- **Method node**  
  - Represents a computation step.  
  - Holds references to its parent **Data nodes** (the required inputs).  
  - Holds references to its child nodes (the outputs it produces).  
  - Includes relevant hyperparameters (e.g., λ in Lasso).  

## How the code work

In this section, we briefly describe how the structure and its components interact, using **SFS-DA** as an illustrative example for clarity.


1. To start, we need to define a pipeline from the API:

```{python}
def SFS_DA() -> Pipeline:
    xs = Data()
    ys = Data()

    xt = Data()
    yt = Data()

    OT = OptimalTransportDA()
    x_tilde, y_tilde = OT.run(xs=xs, ys=ys, xt=xt, yt=yt)

    lasso = LassoFeatureSelection(lambda_=10)
    active_set = lasso.run(x_tilde, y_tilde)
    return Pipeline(
        inputs=(xs, ys, xt, yt),
        output=active_set,
        test_statistic=SFS_DATestStatistic(xs=xs, ys=ys, xt=xt, yt=yt),
    )
```

- `Data()` creates a new **Data node**.  
- `OT = OptimalTransportDA()` instantiates a new **Method node** of type `OptimalTransportDA`.  
- The call `OT.run(xs=xs, ys=ys, xt=xt, yt=yt)` specifies the input Data nodes for the method and produces two new **Data nodes**: `x_tilde` and `y_tilde`.  
- The same applies to `LassoFeatureSelection`, which takes `(x_tilde, y_tilde)` as inputs and generates the **Data node** `active_set`.  
- In the final line, `Pipeline(...)` is constructed:  
  - `inputs` are pointers to the input Data nodes `(xs, ys, xt, yt)`.  
  - `output` is a pointer to the output Data node `active_set`.  
  - This output is the data we condition on during inference. 

2. After defining the pipeline and generating the data, we can execute it and perform selective inference on its result simply by calling the pipeline:

```{python}
selected_features, p_values = my_pipeline(
    inputs=[xs, ys, xt, yt], covariances=[sigma_s, sigma_t]
)
```

3. Method `_call_` of Pipeline:

```{python}
    def __call__(
        self,
        inputs: List[npt.NDArray[np.floating]],
        covariances: List[npt.NDArray[np.floating]],
        verbose: bool = False,
    ) -> npt.NDArray[np.floating]:

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
```

This code performs the following tasks sequentially:

- Updates the input data in the input nodes.  
- Calls `self.output_node()` to recursively compute the output using the input data from the previous step. This relies on the `__call__` method of the output node (which is of type `Data`):

```{python}
    def __call__(self) -> npt.NDArray[np.floating]:
        if self.parent is None: # This means this node is the input data node
            if self.data is None:
                raise ValueError("Data node has no data or parent to compute from.")
            return self.data
        self.parent()  # The parent is a method-typed node, which should automatically update the data in this node
        return self.data
```

This method either calls the parent node to compute and return its value, or directly returns the stored value if the current node is an input data node.  

- In this case, the current data node is `active_set`. Its parent (`self.parent`) is the `LassoFeatureSelection` node, whose `__call__` method is invoked. 

```{python}
    def __call__(self) -> npt.NDArray[np.floating]:
        x = self.x_node()
        y = self.y_node()

        active_set, _, _ = self.forward(x=x, y=y)

        self.active_set_node.update(active_set)
        return active_set
```

This method first retrieves `x` and `y` as inputs (which can be interpreted as the transformed data after OT-DA). These inputs are obtained recursively in the same way as for `active_set`. Next, the `forward` method is used to compute the output of the current method node (`LassoFeatureSelection`), which produces the `active_set`. Finally, the `active_set` is updated in the corresponding `active_set_node` through its `update` method.  

- For each `output_id` in the output (in this case, for each feature in the `active_set`), the pipeline performs inference on that feature using its `inference` method.  

```{python}
    def inference(
        self,
        output_id: int,
        covariances: List[npt.NDArray[np.floating]],
        output: npt.NDArray[np.floating],
    ) -> float:
        test_statistic_direction, a, b, test_statistic, variance, deviation = (
            self.test_statistic(output, output_id, covariances)
        )

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
```

- Compute the test statistic and related quantities for the chosen test. In this case, the test is for feature selection after domain adaptation, and it is implemented in the `__call__` method of `SFS_DATestStatistic`.

```{python}
class SFS_DATestStatistic:
    def __call__(
        self,
        active_set: npt.NDArray[np.floating],
        feature_id: int,
        Sigmas: List[npt.NDArray[np.floating]],
    ) -> Tuple[list, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
        xs = self.xs_node()
        ys = self.ys_node()
        xt = self.xt_node()
        yt = self.yt_node()

        y = np.vstack((ys, yt))

        Sigma_s = Sigmas[0]
        Sigma_t = Sigmas[1]
        Sigma = block_diag(Sigma_s, Sigma_t)

        x_active = xt[:, active_set]
        ej = np.zeros((len(active_set), 1))
        ej[feature_id, 0] = 1
        test_statistic_direction = np.vstack(
            (
                np.zeros((xs.shape[0], 1)),
                x_active.dot(np.linalg.inv(x_active.T.dot(x_active))).dot(ej),
            )
        )

        b = Sigma.dot(test_statistic_direction).dot(
            np.linalg.inv(
                test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)
            )
        )
        a = (
            np.identity(x_active.shape[0] + xs.shape[0])
            - b.dot(test_statistic_direction.T)
        ).dot(y)

        test_statistic = test_statistic_direction.T.dot(y)[0, 0]
        variance = test_statistic_direction.T.dot(Sigma).dot(test_statistic_direction)[
            0, 0
        ]
        deviation = np.sqrt(variance)
        self.xs_node.parametrize(data=xs)
        self.ys_node.parametrize(a=a[: xs.shape[0], :], b=b[: xs.shape[0], :])
        self.xt_node.parametrize(data=xt)
        self.yt_node.parametrize(a=a[xs.shape[0] :, :], b=b[xs.shape[0] :, :])
        return test_statistic_direction, a, b, test_statistic, variance, deviation
```
This method computes the test statistic direction, the parametrized parameters (`a`, `b`), the test statistic itself, and its variance and deviation. It then uses the `parametrize` method of the `Data` class to update the parametrized variables in both `ys_node` and `yt_node`.

- After obtaining the test statistic and related quantities, the `line_search` function is used to find intervals of `z` where the expression `a + b z` matches the observed `active_set`.  
```{python}
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
        assert list_intervals[i][1] <= list_intervals[i + 1][0] + 1e-13, (
            f"Intervals are overlapping in line search: {list_intervals[i]} and {list_intervals[i + 1]}"
        )

    return list_intervals, list_outputs
````

- Inside this function, the `inference` method of `output` (a `Data` node) is called.  
```{python}
class Data:
    def inference(self, z: float):
        if self.parent is not None: # If it has Method parent call its parent's method ```inference``` to achieve interval
            interval = self.parent.inference(z)
        else: # If not, interval is [-np.inf, np.inf]
            interval = [-np.inf, np.inf]

        if self.a is not None and self.b is not None: # If the data in this node is a parametrizable variable
            self.inference_data = self.a + self.b * z
        return self.inference_data, self.a, self.b, interval
```
This method computes the feasible interval for the current value of `z`, propagating from the input node up to the current node and also evaluates the data at that `z`.  

- The `inference` method of `self.parent` (a `Method` node) is then called. For example, in this case we examine the `inference` method of `LassoFeatureSelection`.  
```{python}
class LassoFeatureSelection:
    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            self.active_set_node.parametrize(data=self.active_set_data)
            return self.interval

        x, _, _, interval_x = self.x_node.inference(z)
        y, a, b, interval_y = self.y_node.inference(z)

        active_set, inactive_set, sign_active = self.forward(x, y)
        inactive_set = np.setdiff1d(np.arange(x.shape[1]), active_set)

        self.active_set_node.parametrize(data=active_set)

        # x_a: x with active features
        x_a = x[:, active_set]
        # x_i: x with inactive features
        x_i = x[:, inactive_set]

        x_a_plus = np.linalg.inv(x_a.T.dot(x_a)).dot(x_a.T)
        x_aT_plus = x_a.dot(np.linalg.inv(x_a.T.dot(x_a)))
        temp = x_i.T.dot(x_aT_plus).dot(sign_active)

        # A + Bz <= 0 (elemen-wise)
        A0 = self.lambda_ * sign_active * np.linalg.inv(x_a.T.dot(x_a)).dot(
            sign_active
        ) - sign_active * x_a_plus.dot(a)
        B0 = -1 * sign_active * x_a_plus.dot(b)

        temperal_variable = x_i.T.dot(np.identity(x.shape[0]) - x_a.dot(x_a_plus))

        A10 = -(
            np.ones((temp.shape[0], 1))
            - temp
            - (temperal_variable.dot(a)) / self.lambda_
        )
        B10 = (temperal_variable.dot(b)) / self.lambda_

        A11 = -(
            np.ones((temp.shape[0], 1))
            + temp
            + (temperal_variable.dot(a)) / self.lambda_
        )
        B11 = -(temperal_variable.dot(b)) / self.lambda_

        solve_linear_inequalities(A0, B0)
        solve_linear_inequalities(A10, B10)
        solve_linear_inequalities(A11, B11)

        A = np.vstack((A0, A10, A11))
        B = np.vstack((B0, B10, B11))

        final_interval = intersect(interval_x, interval_y)
        final_interval = intersect(final_interval, solve_linear_inequalities(A, B))

        self.active_set_node.parametrize(data=active_set)

        self.interval = final_interval
        self.active_set_data = active_set

        return final_interval
```
The first `if` statement handles repeated computations: if the input `z` lies within a previously calculated interval, the method simply returns that interval without recalculating. If not, the interval is computed by solving a system of linear inequalities, then intersected with the intervals from its input nodes. Finally, the output data is updated in its corresponding node using the `parametrize` method, the cached interval and output data are refreshed, and the final interval is returned.  
- The same approach applies to earlier data nodes and to `OTDomainAdaptation`.  

## How to add a method

Here how a method should be added:

1. Identify the submodule where the method belongs to or create a new submodule

2. Create a new file with the method name or similar, for example: lasso, optimal_transport,...

3. Create a Class which must include methods:

- `__init__` : init a class
- `forward`: input some data and perform calculation the results for the method, example:

```{python}
def forward(
        self, x: npt.NDArray[np.floating], y: npt.NDArray[np.floating]
    ) -> Tuple[
        npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]
    ]:
        num_of_dimension = x.shape[1]

        lasso = Lasso(
            alpha=self.lambda_ / x.shape[0],
            fit_intercept=False,
            tol=1e-10,
            max_iter=100000000,
        )
        lasso.fit(x, y)

        coefficients = lasso.coef_.reshape(num_of_dimension, 1)
        active_set = np.nonzero(coefficients)[0]
        inactive_set = np.setdiff1d(np.arange(num_of_dimension), active_set)
        sign_active = np.sign(coefficients[active_set]).reshape(-1, 1)

        # # Uncomment this to checkKKT for Lasso
        # self.checkKKT_Lasso(x, y, coefficients, self.lambda_)

        return active_set, inactive_set, sign_active
```

- `run`: Configure input data node and return output data node, example:
```{python}
    def run(self, x: Data, y: Data) -> Data:
        self.x_node = x
        self.y_node = y
        return self.active_set_node
```
- `__call__`: Compute the result by using `forward` and update the computed result to the output data node, examples:
```{python}
    def __call__(self) -> npt.NDArray[np.floating]:
        x = self.x_node()
        y = self.y_node()

        active_set, _, _ = self.forward(x=x, y=y)

        self.active_set_node.update(active_set)
        return active_set
```

-`inference`: Find the feastible interval for the parametrized data at `z` and update the interval and inference data into its output data node, example:
```{python}
    def inference(self, z: float) -> Tuple[list, npt.NDArray[np.floating]]:
        r"""Find feasible interval of the Lasso Feature Selection for the parametrized data at z.

        ----------
        z : float
            Inference parameter value

        Returns
        -------
        final_interval : list
            Feasible interval [lower, upper] for z
        """
        if self.interval is not None and self.interval[0] <= z <= self.interval[1]:
            self.active_set_node.parametrize(data=self.active_set_data)
            return self.interval

        x, _, _, interval_x = self.x_node.inference(z)
        y, a, b, interval_y = self.y_node.inference(z)

        active_set, inactive_set, sign_active = self.forward(x, y)
        inactive_set = np.setdiff1d(np.arange(x.shape[1]), active_set)

        self.active_set_node.parametrize(data=active_set)

        # x_a: x with active features
        x_a = x[:, active_set]
        # x_i: x with inactive features
        x_i = x[:, inactive_set]

        x_a_plus = np.linalg.inv(x_a.T.dot(x_a)).dot(x_a.T)
        x_aT_plus = x_a.dot(np.linalg.inv(x_a.T.dot(x_a)))
        temp = x_i.T.dot(x_aT_plus).dot(sign_active)

        # A + Bz <= 0 (elemen-wise)
        A0 = self.lambda_ * sign_active * np.linalg.inv(x_a.T.dot(x_a)).dot(
            sign_active
        ) - sign_active * x_a_plus.dot(a)
        B0 = -1 * sign_active * x_a_plus.dot(b)

        temperal_variable = x_i.T.dot(np.identity(x.shape[0]) - x_a.dot(x_a_plus))

        A10 = -(
            np.ones((temp.shape[0], 1))
            - temp
            - (temperal_variable.dot(a)) / self.lambda_
        )
        B10 = (temperal_variable.dot(b)) / self.lambda_

        A11 = -(
            np.ones((temp.shape[0], 1))
            + temp
            + (temperal_variable.dot(a)) / self.lambda_
        )
        B11 = -(temperal_variable.dot(b)) / self.lambda_

        solve_linear_inequalities(A0, B0)
        solve_linear_inequalities(A10, B10)
        solve_linear_inequalities(A11, B11)

        A = np.vstack((A0, A10, A11))
        B = np.vstack((B0, B10, B11))

        final_interval = intersect(interval_x, interval_y)
        final_interval = intersect(final_interval, solve_linear_inequalities(A, B))

        self.active_set_node.parametrize(data=active_set)

        self.interval = final_interval
        self.active_set_data = active_set

        return final_interval
```
