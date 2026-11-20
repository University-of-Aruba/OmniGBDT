def create_graph(file_name, tree_index=0, value_list=None):
    """Build a Graphviz graph for one tree in a model dump.

    Args:
        file_name: Path to a dumped model.
        tree_index: Zero-based tree index.
        value_list: Output indices to show in each leaf. Defaults to the first.

    Returns:
        A graph with labeled split edges and leaf values.
    """
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError(
            "create_graph() requires the optional graphviz dependency. "
            "Install it with `pip install omnigbdt[plot]` or `pip install graphviz`."
        ) from exc

    if value_list is None:
        value_list = []

    with open(file_name, "r") as f:
        tree = f.read().split("Booster")
        tree.pop(0)
        tree = tree[tree_index]
        tree = tree.split("\n")[1:]

        info = {}
        for line in tree:
            line = line.strip().split(",")
            if len(line) <= 1: continue
            node = int(line.pop(0))
            if node > 0:
                value = [float(_) for _ in line]
            elif node < 0:
                # parent, left, right, column, threshold
                value = [int(line[0]), int(line[1]), int(line[2]), int(line[3]), float(line[4])]
            info.update({node: value})

    graph = Digraph()

    for node, value in info.items():
        if node > 0:
            indices = value_list if len(value_list) else [0]
            label = "\n".join(f"{value[i]:.4f}" for i in indices)
            graph.node(f"L-{node}", shape="record", label=label)
        else:
            graph.node(f"NL-{node}", label=f"X[{value[3]}] {value[4]:.4f}")
            for child, edge_label in zip(value[1:3], ("<=", ">")):
                if child != 0:
                    prefix = "L" if child > 0 else "NL"
                    graph.edge(f"NL-{node}", f"{prefix}-{child}", label=edge_label)
    return graph
