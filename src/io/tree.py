from typing import List


class Tree:
    def __init__(self, num_nodes: int) -> None:
        self._num_nodes = num_nodes
        self._adj_list = {}
        self._m = 0
        self._out_deg = {}
        self._in_deg = {}

    def add_node(self, v: str) -> None:
        self._adj_list[v] = []
        self._out_deg[v] = 0
        self._in_deg[v] = 0

    def add_edge(self, u: str, v: str, length: float) -> None:
        self._adj_list[u].append((v, length))
        self._m += 1
        self._out_deg[u] += 1
        self._in_deg[v] += 1

    def is_node(self, v: str) -> bool:
        return v in self._adj_list

    def nodes(self) -> List[str]:
        return list(self._adj_list.keys())[:]

    def root(self) -> str:
        roots = [u for u in self._adj_list.keys() if self._in_deg[u] == 0]
        if len(roots) != 1:
            raise Exception(f"Tree should have one root, but found: {roots}")
        return roots[0]

    def __str__(self) -> str:
        res = ""
        res += f"Tree with {self._num_nodes} nodes, and {self._m} edges:\n"
        for u in self._adj_list.keys():
            for (v, length) in self._adj_list[u]:
                res += f"{u} -> {v}: {length}\n"
        return res

    def children(self, u: str) -> List[str]:
        return list(self._adj_list[u])[:]

    def is_leaf(self, u: str) -> bool:
        return self._out_deg[u] == 0


def read_tree(
    tree_path: str,
) -> Tree:
    lines = open(tree_path, "r").read().split("\n")
    try:
        n, s = lines[0].split(" ")
        if s != "nodes":
            raise Exception
        n = int(n)
    except Exception:
        raise Exception(
            f"Tree file: {tree_path} should start with '[num_nodes] nodes'. "
            f"It started with: '{lines[0]}'"
        )
    tree = Tree(n)
    for i in range(1, n + 1, 1):
        v = lines[i]
        tree.add_node(v)
    try:
        m, s = lines[n + 1].split(" ")
        if s != "edges":
            raise Exception
        m = int(m)
    except Exception:
        raise Exception(
            f"Tree file: {tree_path} should have line '[num_edges] edges' at "
            f"position {n + 1}, but it had line: '{lines[n + 1]}'"
        )
    for i in range(n + 2, n + 2 + m, 1):
        try:
            u, v, length = lines[i].split(" ")
            length = float(length)
        except Exception:
            raise Exception(
                f"Tree file: {tree_path} should have line '[u] [v] [length]' at"
                f" position {i}, but it had line: '{lines[i]}'"
            )
        if not tree.is_node(u) or not tree.is_node(v):
            raise Exception(
                f"In Tree file {tree_path}: {u} and {v} should be nodes in the"
                f" tree, but the nodes are: {tree.nodes()}"
            )
        tree.add_edge(u, v, length)
    return tree
