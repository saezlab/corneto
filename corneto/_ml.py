import os
from types import MethodType

from corneto._graph import BaseGraph
from collections import deque

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

def toposort(G):
    # Topological sort using Kahn's algorithm
    in_degree = {v: len(set(G.predecessors(v))) for v in G._get_vertices()}

    # Initialize queue with nodes having zero in-degree
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
            """
            A layer for selecting specific indices along a specified axis.

            Args:
                indexes (list or array): The indices to select.
                axis (int): The axis along which to select the indices.
            """
            super(IndexSelector, self).__init__(**kwargs)
            self.indexes = indexes
            self.axis = axis

        def call(self, inputs):
            """
            Apply index selection along the specified axis.
            """
            return keras.ops.take(inputs, self.indexes, axis=self.axis)

        def compute_output_shape(self, input_shape):
            """
            Compute the output shape after index selection.
            """
            # Convert negative axis to positive
            axis = self.axis if self.axis >= 0 else len(input_shape) + self.axis
            new_shape = list(input_shape)
            new_shape[axis] = len(self.indexes)
            return tuple(new_shape)

        def get_config(self):
            """
            Serialize the configuration of the layer for saving/loading models.
            """
            config = super(IndexSelector, self).get_config()
            config.update({"indexes": self.indexes, "axis": self.axis})
            return config

    return IndexSelector

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
    verbose=False,
):
    keras = _load_keras()
    input_layer = keras.Input(shape=(len(input_nodes),), name="inputs")
    if batch_norm_input:
        input_layer = keras.layers.BatchNormalization(
            center=batch_norm_center, scale=batch_norm_scale
        )(input_layer)
    if unit_norm_input:
        input_layer = keras.layers.UnitNormalization()(input_layer)
    vertices = G.toposort()
    input_index = {v: i for i, v in enumerate(input_nodes)}
    kernel_reg, bias_reg = None, None
    if kernel_reg_l1 > 0 or kernel_reg_l2 > 0:
        kernel_reg = keras.regularizers.l1_l2(
            l1=bias_reg_l1, l2=bias_reg_l2
        )
    if bias_reg_l1 > 0 or bias_reg_l2 > 0:
        bias_reg = keras.regularizers.l1_l2(
            l1=kernel_reg_l1, l2=kernel_reg_l2
        )
    neurons = {}
    for v in vertices:
        neuron_inputs = []
        in_v_inputs = []
        in_v_neurons = []
        predecessors = list(G.predecessors(v))
        if len(predecessors) == 0:
            continue
        for par in predecessors:
            if par in input_index:
                in_v_inputs.append(input_index[par])
            else:
                in_v_neurons.append(par)
        # Collect inputs from the input layer
        if len(in_v_inputs) > 1:
            #inputs = _concat_indexes(input_layer, in_v_inputs, keras=k)
            inputs = index_selector()(in_v_inputs)(input_layer)
            if dropout > 0 and len(in_v_inputs) >= min_inputs_for_dropout:
                inputs = keras.layers.Dropout(dropout)(inputs)
            neuron_inputs.append(inputs)
        elif len(in_v_inputs) == 1:
            j = in_v_inputs[0]
            neuron_inputs.append(input_layer[:, j : (j+1)])
        # Collect inputs from other neurons
        if len(in_v_neurons) > 1:
            inputs = keras.layers.Concatenate()(
                [neurons[p] for p in in_v_neurons]
            )
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
        default_act = (
            default_hidden_activation
            if v not in output_nodes
            else default_output_activation
        )
        act = G.get_attr_vertex(v).get(activation_attribute, default_act)
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
            print(
                f"Creating {v} ({act}), {len(in_v_inputs)} data input(s), {len(in_v_neurons)} neuron input(s)"
            )
    # Create the model
    if len(output_nodes) == 1:
        output_layer = neurons[output_nodes[0]]
    else:
        output_layer = keras.layers.Concatenate(name="output_layer")(
            [neurons[v] for v in output_nodes]
        )
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def plot_model(model):
    return _load_keras().utils.plot_model(model)
