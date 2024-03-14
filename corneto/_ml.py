from corneto._graph import BaseGraph
import numpy as np
import os
import sys


def _load_keras():
    # Check if keras_core is installed,
    # then find the best backend, prioritizing JAX then TF then Pytorch:
    try:
        keras_env = os.environ.get("KERAS_BACKEND", None)
        if keras_env is None:
            keras_env = "jax"
            os.environ["KERAS_BACKEND"] = keras_env
        import keras_core as keras

        return keras
    except ImportError as e:
        raise e


def _concat_indexes(layer, indexes, keras):
    if len(indexes) > 1:
        if len(set(indexes)) == layer.shape[1]:
            subset = layer
        else:
            slices = [layer[:, j : (j + 1)] for j in indexes]
            subset = keras.layers.Concatenate()(slices)
    else:
        j = list(indexes)[0]
        subset = layer[:, j : (j + 1)]
    return subset


def create_dagnn(
    G: BaseGraph,
    input_nodes,
    output_nodes,
    kernel_reg=None,
    bias_reg=None,
    verbose=False,
    activation_attribute="activation",
    default_hidden_activation="relu",
    default_output_activation="linear",
):
    keras = _load_keras()
    input_layer = keras.Input(shape=(len(input_nodes),), name="inputs")
    input_index = {v: i for i, v in enumerate(input_nodes)}
    queue = list(input_nodes)
    neurons = {}
    concat_cache = {}
    while len(queue) > 0:
        v = queue.pop(0)
        for s in G.successors(v):
            if s not in neurons:
                queue.append(s)
            else:
                continue
            if s not in input_index:
                n_inputs = []
                s_idx_inputs = set()
                s_neu_inputs = set()
                for p in G.predecessors(s):
                    if p in input_index:
                        idx = input_index[p]
                        s_idx_inputs.add(idx)
                    else:
                        s_neu_inputs.add(p)
                # Check if all neuron inputs are created
                if len(s_neu_inputs) > 0:
                    if not all([p in neurons for p in s_neu_inputs]):
                        continue
                # Now check if there is a cached concatenation
                # for the inputs of this neuron
                if len(s_idx_inputs) > 0:
                    s_idx_inputs = frozenset(s_idx_inputs)
                    if s_idx_inputs in concat_cache:
                        # print(f"  > Using cached concatenation, {s_idx_inputs}")
                        n_inputs.append(concat_cache[s_idx_inputs])
                    else:
                        subset_inputs = _concat_indexes(
                            input_layer, s_idx_inputs, keras
                        )
                        concat_cache[s_idx_inputs] = subset_inputs
                        n_inputs.append(subset_inputs)
                if len(s_neu_inputs) > 0:
                    s_neu_inputs = frozenset(s_neu_inputs)
                    if s_neu_inputs in concat_cache:
                        n_inputs.append(concat_cache[s_neu_inputs])
                    else:
                        if len(s_neu_inputs) > 1:
                            subset_inputs = keras.layers.Concatenate()(
                                [neurons[p] for p in s_neu_inputs]
                            )
                            concat_cache[s_neu_inputs] = subset_inputs
                        else:
                            subset_inputs = neurons[list(s_neu_inputs)[0]]
                        n_inputs.append(subset_inputs)
                if len(n_inputs) > 1:
                    neuron_inputs = keras.layers.Concatenate(name=f"{s}_c")(n_inputs)
                else:
                    neuron_inputs = n_inputs[0]

                # Create the neuron.
                default_act = (
                    default_hidden_activation
                    if s not in output_nodes
                    else default_output_activation
                )
                act = G.get_attr_vertex(s).get(activation_attribute, default_act)
                neuron = keras.layers.Dense(
                    1,
                    activation=act,
                    kernel_regularizer=kernel_reg,
                    bias_regularizer=bias_reg,
                    name=s,
                )
                x = neuron(neuron_inputs)
                neurons[s] = x
                if verbose:
                    print(
                        f"{s} ({act}) > {len(s_idx_inputs)} data input(s), {len(s_neu_inputs)} neuron input(s)"
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


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    # from sklearn
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        idx = np.random.permutation(len(X))
    else:
        idx = np.arange(len(X))

    # Calculate the test set size
    test_set_size = int(len(X) * test_size)

    # Split the indices for the train and test set
    test_indices = idx[:test_set_size]
    train_indices = idx[test_set_size:]

    # Split the data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
