import warnings
from typing import Any, Dict, Iterable, Optional

from corneto._legacy import ReNet
from corneto._types import StrOrInt

# TODO: Pass default style to plotting methods
_default_style = {
    "edges": {
        "style": "solid",
        "node_size": 1400,
        "arrowstyle": "-|>",
        "edge_color": "black",
        "alpha": 1.0,
        "width": 1.0,
        "connectionstyle": "arc3, rad = 0.1",
    },
    "nodes": {"size": 800, "color": "white", "edgecolor": "black"},
}


def style(edge_rad=0.1, node_size=800, node_margin_factor=0.80, edge_alpha=1.0):
    st = _default_style
    # TODO: config more styles
    n_size = int(node_size * (1 + node_margin_factor))
    st["edges"]["node_size"] = n_size
    st["edges"]["connection_style"] = f"arc3, rad = {edge_rad}"
    st["edges"]["alpha"] = edge_alpha
    st["nodes"]["size"] = node_size
    return st


def stylize(variables):
    edge_props, node_props = {}, {}
    up_edges = [e for e, v in variables["reactions"].items() if v[0].x > 0]
    down_edges = [e for e, v in variables["reactions"].items() if v[1].x > 0]
    up_nodes = [n for n, v in variables["species"].items() if v.x > 0]
    down_nodes = [n for n, v in variables["species"].items() if v.x < 0]
    for e in up_edges:
        edge_props[e] = {"edge_color": "tab:red", "width": 2.0}
    for e in down_edges:
        edge_props[e] = {"edge_color": "tab:blue", "width": 2.0}
    for n in up_nodes:
        node_props[n] = {"color": "tab:red"}
    for n in down_nodes:
        node_props[n] = {"color": "tab:blue"}
    return {"nodes": node_props, "edges": edge_props}


def get_color(d, default="black"):
    edgecolor = default
    if "weight" in d:
        if d["weight"] > 0:
            edgecolor = "tab:red"
        elif d["weight"] < 0:
            edgecolor = "tab:blue"
        else:
            edgecolor = "grey"
    return edgecolor


def get_style(s, d):
    style = "solid"
    if s.startswith("_") or ("weight" in d and d["weight"] == 0):
        style = "--"
    return style


def plot(
    G,
    pos=None,
    node_size=800,
    margin_factor=0.80,
    edge_rad=0.1,
    edge_alpha=1.0,
    custom_style=None,
    ax=None,
    node_labels=None,
    figsize=None,
):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.patches import ArrowStyle
    except ImportError:
        raise ImportError("matplotlib and networkx are required for plotting")

    fig = None
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    if pos is None:
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception as err:
            warnings.warn(f"Failed to use graphviz with dot layout: {err!s}. Using spring_layout instead.")
            pos = nx.spring_layout(G)

    inhibitor_style = ArrowStyle("-[", widthB=1.0, lengthB=0.0, angleB=None)
    n_size = int(node_size * (1 + margin_factor))

    # Collect the properties for edges/nodes before drawing
    node_props, edge_props = {}, {}
    for n, d in G.nodes(data=True):
        node_props[n] = {
            "size": node_size,
            "color": "white",
            "edgecolor": get_color(d),
            "node_shape": "o",
        }
        if "type" in d and d["type"] == "reaction":
            node_props[n]["node_shape"] = "s"
    for s, t, d in G.edges(data=True):
        style = "--" if "type" in d and d["type"] == "simple" else "solid"
        edge_props[(s, t)] = {
            "style": style,
            "node_size": n_size,
            "arrowstyle": "-|>",
            "edge_color": "black",
            "alpha": edge_alpha,
            "width": 1.0,
            "connectionstyle": f"arc3, rad = {edge_rad}",
        }

    for s, t, d in G.edges(data=True):
        if "weight" in d and float(d["weight"]) < 0:
            edge_props[(s, t)]["arrowstyle"] = inhibitor_style
        if s.startswith("_") or ("type" in d and d["type"] == "dummy"):
            edge_props[(s, t)]["style"] = "--"

    # --- Apply custom properties ---
    if custom_style and "edges" in custom_style:
        for edge, properties in custom_style["edges"].items():
            if edge in edge_props:
                edge_props[edge].update(properties)
            else:
                # Check by id
                for s, t, d in G.edges(data=True):
                    if d["id"] == edge:
                        edge_props[(s, t)].update(properties)
    if custom_style and "nodes" in custom_style:
        for node, properties in custom_style["nodes"].items():
            if node in node_props:
                node_props[node].update(properties)

    # --- Draw everything ---
    # TODO: inject ax into node props and pass **node_props[n] to draw_networkx_nodes
    for n in G.nodes():
        nx_node = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[n],
            node_size=node_props[n]["size"],
            node_color=node_props[n]["color"],
            node_shape=node_props[n]["node_shape"],
            ax=ax,
        )
        nx_node.set_edgecolor(node_props[n]["edgecolor"])

    for e in G.edges():
        nx_edge = nx.draw_networkx_edges(
            G,
            pos,
            # just pass **edge_props[e], change name of params
            # that do not match
            style=edge_props[e]["style"],
            width=edge_props[e]["width"],
            arrowstyle=edge_props[e]["arrowstyle"],
            edge_color=edge_props[e]["edge_color"],
            node_size=edge_props[e]["node_size"],
            alpha=edge_props[e]["alpha"],
            connectionstyle=edge_props[e]["connectionstyle"],
            edgelist=[e],
            ax=ax,
        )
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax)
    return fig, ax, pos


def to_nxgraph(
    renet: ReNet,
    # reactions: Optional[Union[Iterable[str], Iterable[int]]] = None
    reactions: Optional[Iterable[StrOrInt]] = None,
):
    try:
        from networkx import DiGraph, set_edge_attributes, set_node_attributes
    except ImportError:
        raise ImportError("NetworkX is required to convert a Reaction Network to a networkx graph.")
    G = DiGraph()
    edges = []
    edge_attributes = dict()
    if reactions is None:
        rxns = list(range(len(renet.reactions)))
    else:
        rxns = [renet.get_reaction_id(r) if isinstance(r, str) else r for r in reactions]
    for rid in rxns:
        reactants, products = (
            renet.get_reactants_of_reaction(rid),
            renet.get_products_of_reaction(rid),
        )
        if len(reactants) == 0 or len(products) == 0:
            # ignore import/export reactions? (with no reactant or product?)
            continue
        elif len(reactants) == len(products) == 1:
            (r,) = reactants
            (p,) = products
            edges.append(
                (
                    renet.species[r],
                    renet.species[p],
                    renet.properties.reaction_value(rid),
                )
            )
            # TODO: Change this attribute name
            edge_attributes[(renet.species[r], renet.species[p])] = {
                "type": "simple",
                "id": rid,
                "name": renet.reactions[rid],
            }
        else:
            # Hyperedge (reactants and products are connected through an intermediate node)
            # print(reactants, products)
            prop = {"type": "complex", "id": rid, "name": renet.reactions[rid]}
            for r in reactants:
                coeff = renet.stoichiometry[r, rid]
                # put 0 otherwise it draws activation/inhibition
                coeff = 0
                edges.append((renet.species[r], renet.reactions[rid], coeff))
                edge_attributes[(renet.species[r], renet.reactions[rid])] = prop
            for p in products:
                coeff = renet.stoichiometry[p, rid]
                edges.append((renet.reactions[rid], renet.species[p], coeff))
                edge_attributes[(renet.reactions[rid], renet.species[p])] = prop
    G.add_weighted_edges_from(edges)
    set_edge_attributes(G, edge_attributes)
    node_attributes: Dict[str, Any] = dict()
    for n in G.nodes():
        n_id = str(n)
        attributes: Dict[str, Any] = dict()
        # check if n_id is in _compounds_ids list, and return the position in the list
        id = None
        if n_id in renet.species:
            id = renet.get_species_id(n_id)
        if id is not None:
            attributes["weight"] = renet.properties.species_value(n_id)
            attributes["type"] = "compound"
        else:
            attributes["type"] = "reaction"
        if n_id.startswith("_"):
            attributes["type"] = "dummy"
        node_attributes[n_id] = attributes
    set_node_attributes(G, node_attributes)
    return G
