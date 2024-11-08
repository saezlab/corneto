import random

from corneto._graph import Graph


class TimeSuite:
    def setup(self):
        # Create a graph with at least 1000 edges
        self.g = Graph()
        self.num_vertices = 500
        self.num_edges = 5000

        # Randomly generate edges for the graph
        for _ in range(self.num_edges):
            v1 = random.randint(0, self.num_vertices - 1)
            v2 = random.randint(0, self.num_vertices - 1)
            self.g.add_edge(v1, v2)

    def time_add_single_vertex(self):
        self.g = Graph()
        for _ in range(1000):
            self.g.add_vertex("v")

    def time_add_single_edge(self):
        self.g = Graph()
        for i in range(1000):
            self.g.add_edge(f"a{i}", f"b{i}")

    def time_add_edges(self):
        self.g = Graph()
        edges = [(i, i + 1) for i in range(1000)]
        self.g.add_edges(edges)

    def time_bfs(self):
        self.g.bfs(self.g.V[0])

    def time_bfs_rev(self):
        self.g.bfs(self.g.V[0], reverse=True)

    def time_in_edges(self):
        list([self.g.in_edges(self.g.V[i]) for i in range(self.g.num_vertices)])

    def time_out_edges(self):
        list([self.g.out_edges(self.g.V[i]) for i in range(self.g.num_vertices)])
