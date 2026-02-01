import os
from collections import deque

import numpy as np

from corneto.graph import Attr, EdgeType

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def _load_keras():
    # Check if keras_core is installed,
    # then find the best backend, prioritizing JAX then TF then Pytorch:
    try:
        keras_env = os.environ.get("KERAS_BACKEND", None)
        if keras_env is None:
            keras_env = "jax"
            os.environ["KERAS_BACKEND"] = keras_env
        import keras

        return keras
    except ImportError as e:
        raise e


def kfold_nonzero_splits(data, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
    from sklearn.model_selection import KFold

    """
    Perform K-fold splitting on all nonzero cells in the input data,
    returning two structures per fold (train and val), each the same shape
    as the original data, but with NaNs (or a numpy.nan equivalent) marking
    the 'left-out' entries.

    Parameters
    ----------
    data : array-like or pd.DataFrame
        Input data with shape (features x samples) and values in {-1, 0, 1}.
        Can be a numpy array or a pandas DataFrame.
    n_splits : int, optional
        Number of folds (default=5).
    shuffle : bool, optional
        Shuffle the labeled cells before splitting (default=True).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns
    ------
    train, val : tuple of the same type as `data`
        - Both have the same shape as `data`.
        - In `train`, all cells that belong to the validation fold are set to NaN.
        - In `val`, all cells that are not in the validation fold are set to NaN.
    """
    if _PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
        is_pandas = True
        arr = data.to_numpy()
    else:
        is_pandas = False
        arr = np.asarray(data)

    # Identify positions of nonzero (±1) cells
    row_indices, col_indices = np.where(arr != 0)
    labeled_positions = np.array(list(zip(row_indices, col_indices)))

    # Use KFold to split these labeled positions
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_idx, val_idx in kf.split(labeled_positions):
        train_copy = arr.copy()
        val_copy = arr.copy()

        # Set NaNs for validation fold in train_copy
        val_positions = labeled_positions[val_idx]
        for r, c in val_positions:
            train_copy[r, c] = np.nan

        # Set NaNs for training fold in val_copy
        train_positions = labeled_positions[train_idx]
        for r, c in train_positions:
            val_copy[r, c] = np.nan

        if is_pandas:
            train_copy = pd.DataFrame(train_copy, index=data.index, columns=data.columns)
            val_copy = pd.DataFrame(val_copy, index=data.index, columns=data.columns)

        yield train_copy, val_copy


def toposort(G):
    # Topological sort using Kahn's algorithm
    # See: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_sort.html
    in_degree = {v: len(set(G.predecessors(v))) for v in G._get_vertices()}

    # Initialize queue with nodes having zero in-degrees
    queue = deque([v for v in in_degree.keys() if in_degree[v] == 0])

    result = []

    while queue:
        v = queue.popleft()
        result.append(v)

        # Decrease the in-degree of successor nodes by 1
        for successor in G.successors(v):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    # Check if topological sort is possible (i.e., graph has no cycles)
    if len(result) == G.num_vertices:
        return result
    else:
        raise ValueError("Graph contains a cycle, so topological sort is not possible.")


def index_selector():
    keras = _load_keras()

    # Dynamically create IndexSelector class
    @keras.saving.register_keras_serializable(package="Custom")
    class IndexSelector(keras.layers.Layer):
        def __init__(self, indexes, axis=-1, **kwargs):
            """A layer for selecting specific indices along a specified axis.

            Args:
                indexes (list or array): The indices to select.
                axis (int): The axis along which to select the indices.
            """
            super(IndexSelector, self).__init__(**kwargs)
            self.indexes = indexes
            self.axis = axis

        def call(self, inputs):
            """Apply index selection along the specified axis."""
            return keras.ops.take(inputs, self.indexes, axis=self.axis)

        def compute_output_shape(self, input_shape):
            """Compute the output shape after index selection."""
            # Convert negative axis to positive
            axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis
            new_shape = list(input_shape)
            new_shape[axis] = len(self.indexes)
            return tuple(new_shape)

        def get_config(self):
            """Serialize the configuration of the layer for saving/loading models."""
            config = super(IndexSelector, self).get_config()
            config.update({"indexes": self.indexes, "axis": self.axis})
            return config

    return IndexSelector


def _coerce_interaction_sign(value, default=1.0):
    if value is None:
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    return 1.0 if v >= 0 else -1.0


def _edge_interaction_sign(G, source, target, interaction_attribute="interaction", default=1.0):
    for idx, (s, t) in G.edges(vertices=[target]):
        attr = G.get_attr_edge(idx)
        etype = attr.get_attr(Attr.EDGE_TYPE, EdgeType.DIRECTED.value)
        if etype == EdgeType.UNDIRECTED.value:
            continue
        if (source == s or source in s) and (target == t or target in t):
            return _coerce_interaction_sign(attr.get(interaction_attribute, default), default=default)
    return _coerce_interaction_sign(default, default=default)


def signed_dense():
    keras = _load_keras()

    @keras.saving.register_keras_serializable(package="Custom")
    class SignedDense(keras.layers.Layer):
        def __init__(
            self,
            units,
            kernel_sign,
            activation=None,
            kernel_regularizer=None,
            bias_regularizer=None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.units = int(units)
            self.kernel_sign = kernel_sign
            self.activation = keras.activations.get(activation)
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
            self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        def build(self, input_shape):
            input_dim = int(input_shape[-1])
            ks = np.asarray(self.kernel_sign, dtype="float32")
            if ks.ndim == 1:
                if ks.shape[0] != input_dim:
                    raise ValueError("kernel_sign length must match input dimension")
                ks = ks.reshape((input_dim, 1))
            elif ks.ndim == 2:
                if ks.shape[0] != input_dim:
                    raise ValueError("kernel_sign first dimension must match input dimension")
                if ks.shape[1] != self.units:
                    raise ValueError("kernel_sign second dimension must match units")
            else:
                raise ValueError("kernel_sign must be a 1D or 2D array")

            self.kernel_sign_tensor = keras.ops.convert_to_tensor(ks)
            self.kernel_raw = self.add_weight(
                name="kernel_raw",
                shape=(input_dim, self.units),
                initializer="glorot_uniform",
                regularizer=self.kernel_regularizer,
                trainable=True,
            )
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                regularizer=self.bias_regularizer,
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            kernel = self.kernel_sign_tensor * keras.ops.softplus(self.kernel_raw)
            outputs = keras.ops.matmul(inputs, kernel) + self.bias
            if self.activation is not None:
                return self.activation(outputs)
            return outputs

        def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "units": self.units,
                    "kernel_sign": np.asarray(self.kernel_sign, dtype="float32").tolist(),
                    "activation": keras.activations.serialize(self.activation),
                    "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
                    "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
                }
            )
            return config

    return SignedDense


def build_dagnn(
    G,
    input_nodes,
    output_nodes,
    bias_reg_l1=0,
    bias_reg_l2=0,
    kernel_reg_l1=0,
    kernel_reg_l2=0,
    batch_norm_input=True,
    batch_norm_center=False,
    batch_norm_scale=False,
    unit_norm_input=False,
    dropout=0.20,
    min_inputs_for_dropout=2,
    activation_attribute="activation",
    default_hidden_activation="sigmoid",
    default_output_activation="sigmoid",
    force_sign=True,
    interaction_attribute="interaction",
    verbose=False,
):
    keras = _load_keras()
    input_layer = keras.Input(shape=(len(input_nodes),), name="inputs")
    if batch_norm_input:
        input_layer = keras.layers.BatchNormalization(center=batch_norm_center, scale=batch_norm_scale)(input_layer)
    if unit_norm_input:
        input_layer = keras.layers.UnitNormalization()(input_layer)
    vertices = G.toposort()
    input_index = {v: i for i, v in enumerate(input_nodes)}
    kernel_reg, bias_reg = None, None
    if kernel_reg_l1 > 0 or kernel_reg_l2 > 0:
        kernel_reg = keras.regularizers.l1_l2(l1=bias_reg_l1, l2=bias_reg_l2)
    if bias_reg_l1 > 0 or bias_reg_l2 > 0:
        bias_reg = keras.regularizers.l1_l2(l1=kernel_reg_l1, l2=kernel_reg_l2)
    SignedDense = signed_dense() if force_sign else None
    neurons = {}
    for v in vertices:
        neuron_inputs = []
        in_v_inputs = []
        in_v_inputs_idx = []
        in_v_neurons = []
        predecessors = list(G.predecessors(v))
        if len(predecessors) == 0:
            continue
        pred_signs = {}
        for par in predecessors:
            pred_signs[par] = _edge_interaction_sign(
                G,
                par,
                v,
                interaction_attribute=interaction_attribute,
                default=1.0,
            )
            if par in input_index:
                in_v_inputs.append(par)
                in_v_inputs_idx.append(input_index[par])
            else:
                in_v_neurons.append(par)
        # Collect inputs from the input layer
        if len(in_v_inputs_idx) > 1:
            # inputs = _concat_indexes(input_layer, in_v_inputs, keras=k)
            inputs = index_selector()(in_v_inputs_idx)(input_layer)
            if dropout > 0 and len(in_v_inputs_idx) >= min_inputs_for_dropout:
                inputs = keras.layers.Dropout(dropout)(inputs)
            neuron_inputs.append(inputs)
        elif len(in_v_inputs_idx) == 1:
            j = in_v_inputs_idx[0]
            neuron_inputs.append(input_layer[:, j : (j + 1)])
        # Collect inputs from other neurons
        if len(in_v_neurons) > 1:
            inputs = keras.layers.Concatenate()([neurons[p] for p in in_v_neurons])
            if dropout > 0 and len(in_v_neurons) >= min_inputs_for_dropout:
                inputs = keras.layers.Dropout(dropout)(inputs)
            neuron_inputs.append(inputs)
        elif len(in_v_neurons) == 1:
            neuron_inputs.append(neurons[in_v_neurons[0]])
        n_inputs = len(neuron_inputs)
        if n_inputs > 1:
            neuron_inputs = keras.layers.Concatenate(name=f"{v}_concat")(neuron_inputs)
            if dropout > 0 and n_inputs >= min_inputs_for_dropout:
                neuron_inputs = keras.layers.Dropout(dropout)(neuron_inputs)
        else:
            neuron_inputs = neuron_inputs[0]

        # Create the neuron for the current vertex
        default_act = default_hidden_activation if v not in output_nodes else default_output_activation
        act = G.get_attr_vertex(v).get(activation_attribute, default_act)
        if force_sign:
            sign_vector = [pred_signs[p] for p in in_v_inputs] + [pred_signs[p] for p in in_v_neurons]
            neuron = SignedDense(
                1,
                kernel_sign=np.asarray(sign_vector, dtype="float32"),
                activation=act,
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                name=v,
            )
        else:
            neuron = keras.layers.Dense(
                1,
                activation=act,
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                name=v,
            )
        x = neuron(neuron_inputs)
        neurons[v] = x
        if verbose:
            print(f"Creating {v} ({act}), {len(in_v_inputs)} data input(s), {len(in_v_neurons)} neuron input(s)")
    # Create the model
    if len(output_nodes) == 1:
        output_layer = neurons[output_nodes[0]]
    else:
        output_layer = keras.layers.Concatenate(name="output_layer")([neurons[v] for v in output_nodes])
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def plot_model(model):
    return _load_keras().utils.plot_model(model)
