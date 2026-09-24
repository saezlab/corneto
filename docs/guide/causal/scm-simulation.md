# Simulating structural causal models

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

Use `do` for hard clamps, `shift` for known additive offsets, and `replace`
for a general node change. These operations return new models. To compare an
observational and an interventional model using the same exogenous values,
draw noise once and pass it to both:

```python
noise = truth.draw_noise(20, rng=31)
observed = truth.evaluate(noise)
treated = truth.do({"A": 2.0}).evaluate(noise)
```

`to_data` and `sample_data` produce CORNETO `Data` with hard interventions or
known shifts annotated on the affected features. A general `replace` cannot be
represented by `LinearDAGDiscovery`'s intervention metadata, so conversion
raises an error for such models instead of labeling them as hard clamps or
shifts.
