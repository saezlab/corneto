# AnnNet

[AnnNet](https://saezlab.github.io/annnet/) is an annotated network container
with support for directed graphs, hypergraphs, parallel edges, and rich
annotations. CORNETO provides lightweight conversion helpers so the same
network can be used for optimization in CORNETO and annotation or
interoperability workflows in AnnNet.

Install the optional dependency with a concrete AnnNet annotation backend:

```bash
pip install "corneto[annnet]"
```

The converters import AnnNet only when called, so AnnNet remains optional for
all other CORNETO use cases.

## Convert a CORNETO graph

```python
import corneto as cn
from corneto.contrib.annnet import from_annnet, to_annnet

graph = cn.Graph(name="signaling")
graph.add_edge("A", "B", interaction="activation")
graph.add_edge(
    {"A": -2.0, "B": -1.0},
    {"C": 3.0},
    interaction="reaction",
)

annotated = to_annnet(graph)
restored = from_annnet(annotated)
```

Directed binary edges and hyperedges, parallel edges, endpoint coefficients,
and ordinary graph, vertex, and edge attributes are preserved. CORNETO edge
indices become AnnNet IDs such as `corneto_edge_0`. When converting in the
other direction, the original AnnNet ID is stored as the CORNETO edge
attribute `_annnet_edge_id`.

## Scope and limitations

The converters are intentionally small and target the flat, directed
hypergraphs normally used by CORNETO:

- AnnNet layers, slice membership, edge-entities, and flexible direction
  policies are not reproduced in CORNETO.
- Vertex identifiers are converted to strings for AnnNet. A collision after
  conversion raises an error.
- CORNETO edges with an empty source or target set are not supported.
- An undirected hyperedge is converted as one member set. Its original
  CORNETO source/target partition cannot be recovered; converting it back uses
  the full member set on both sides.
- Attributes whose names are reserved for AnnNet structure are omitted with a
  warning. Attribute values must be supported by the installed AnnNet
  dataframe backend.

These limitations do not affect the directed graphs and directed hypergraphs
used by the main CORNETO workflows.
