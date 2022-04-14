from typing import List, Tuple


class Tree:
    def __init__(self) -> None:
        self._num_nodes = 0
        self._num_edges = 0
        self._adj_list = {}
        self._edges = []
        self._m = 0
        self._out_deg = {}
        self._in_deg = {}
        self._parent = {}

    def add_node(self, v: str) -> None:
        self._adj_list[v] = []
        self._out_deg[v] = 0
        self._in_deg[v] = 0
        self._num_nodes += 1

    def add_nodes(self, nodes: List[str]) -> None:
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u: str, v: str, length: float) -> None:
        if v in self._parent:
            raise Exception(
                f"Node {v} already has a parent, graph is not a tree."
            )
        self._adj_list[u].append((v, length))
        self._edges.append((u, v, length))
        self._m += 1
        self._out_deg[u] += 1
        self._in_deg[v] += 1
        self._parent[v] = (u, length)
        self._num_edges += 1

    def add_edges(self, edges: List[Tuple[str, str, float]]) -> None:
        for (u, v, length) in edges:
            self.add_edge(u, v, length)

    def edges(self) -> List[Tuple[str, str, float]]:
        return self._edges[:]

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

    def is_root(self, u: str) -> bool:
        return self._in_deg[u] == 0

    def num_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return self._num_edges

    def preorder_traversal(self) -> List[str]:
        res = []

        def dfs(v: str):
            res.append(v)
            for (u, _) in self.children(v):
                dfs(u)

        dfs(self.root())
        return res

    def parent(self, u: str) -> Tuple[str, int]:
        return self._parent[u]

    def leaves(self) -> List[str]:
        return [u for u in self.nodes() if self.is_leaf(u)]

    def internal_nodes(self) -> List[str]:
        return [u for u in self.nodes() if not self.is_leaf(u)]


def write_tree(
    tree: Tree,
    tree_path: str,
) -> None:
    res = ""
    res += f"{tree.num_nodes()} nodes\n"
    for node in tree.nodes():
        res += f"{node}\n"
    res += f"{tree.num_edges()} edges\n"
    for (u, v, d) in tree.edges():
        res += f"{u} {v} {d}\n"
    open(tree_path, "w").write(res)


def read_tree(
    tree_path: str,
) -> Tree:
    lines = open(tree_path, "r").read().strip().split("\n")
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
    tree = Tree()
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
    if len(lines) != n + 1 + m + 1:
        raise Exception(
            f"Tree file: {tree_path} should have {m} edges, but it has "
            f"{len(lines) - n - 2} edges instead."
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
    assert(tree.num_nodes() == n)
    assert(tree.num_edges() == m)
    return tree
