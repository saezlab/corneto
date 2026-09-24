# Simulating structural causal models

For the runnable version with CORNETO graph diagrams, simulation plots, and a
pandas view of intervention data, see the [SCM simulation notebook](scm-simulation.ipynb).

`SCM` executes one structural equation per variable in a directed acyclic
graph. Each `Node` declares its parents, a callable mechanism, and a callable
noise distribution. Parent columns follow the declared parent order, while
sample output columns follow the variable order in the node mapping.

```python
import numpy as np

from corneto.methods.causal.simulation import Additive, Node, Normal, SCM

scm = SCM(
    {
        "A": Node((), lambda parents, noise: noise, Normal()),
        "B": Node(("A",), lambda parents, noise: 2 * parents[:, 0] + noise, Normal(std=0.5)),
        "C": Node(
            ("A", "B"),
            Additive(lambda parents: np.tanh(parents[:, 0] * parents[:, 1])),
            Normal(std=0.2),
        ),
    }
)
observations = scm.sample(100, rng=7)
```

Mechanisms and noise distributions are ordinary callables, so experiments can
use custom biological equations or noise models without extending the
simulator. A mechanism receives a parent array of shape `(n, number_of_parents)`
and a noise array of shape `(n,)`; it returns one numeric value per sample.
`Linear` and `Additive` provide small helpers for common equations.

Use `Node.replace()` and `SCM.replace()` to customize the baseline model before
sampling. A node edit preserves fields you do not pass, and it does not mark
the resulting data as interventional:

```python
from corneto.methods.causal.simulation import Linear

custom_c = scm.nodes["C"].replace(mechanism=Linear((0.5, 0.25)), noise=Normal(std=0.1))
custom_baseline = scm.replace({"C": custom_c})
custom_data = custom_baseline.sample_data(100, rng=8)
```

`replace()` returns a new model and leaves `scm` unchanged. CORNETO `Data` from
`custom_baseline` is an ordinary sample from that edited baseline.

CORNETO graphs can define dependencies directly. `from_graph` requires a
mechanism for every vertex; isolated vertices become root nodes. It accepts
simple directed edges and rejects cycles, hyperedges, undirected edges,
self-loops, parallel edges, and boundary edges.

```python
from corneto.graph import Graph
from corneto.methods.causal.simulation import Normal, linear_scm

graph = Graph.from_tuples([("A", 1, "B"), ("A", 1, "C"), ("B", 1, "C")])
truth = linear_scm(graph, rng=12, noise=Normal(std=0.1))
data = truth.sample_data(20, rng=25)
```

`linear_scm` returns an ordinary `SCM`. By default it assigns signed random
coefficients whose magnitudes stay away from zero, zero biases, and normal
noise. Pass `weights={(parent, child): value}` and `biases={variable: value}`
for an explicit model. The optional `weight_attribute="coefficient"` reads
that edge attribute; CORNETO's `interaction` sign is not interpreted as a
coefficient magnitude. Its `rng` controls model generation; `sample` and
`sample_data` take their own random state for observations.

Use `do` for hard clamps and `shift` for known additive offsets. Use
`intervene()` for an experimental replacement of an arbitrary node equation;
that operation marks the changed equation as an intervention. These operations
return new models. To compare an observational and an interventional model
using the same exogenous values, draw noise once and pass it to both:

```python
noise = truth.draw_noise(20, rng=31)
observed = truth.evaluate(noise)
treated = truth.do({"A": 2.0}).evaluate(noise)
```

An arbitrary mechanism intervention can be evaluated in the simulator, but it
cannot be represented by `LinearDAGDiscovery`'s intervention metadata:

```python
mechanism_intervention = scm.intervene({"C": custom_c})
experimental_values = mechanism_intervention.sample(100, rng=9)
```

`to_data` and `sample_data` produce CORNETO `Data` with hard interventions or
known shifts annotated on the affected features. `intervene()` models other
experimental changes, but conversion raises an error rather than labeling
them as hard clamps or shifts. Baseline changes made with `replace()` remain
exportable as ordinary data.
