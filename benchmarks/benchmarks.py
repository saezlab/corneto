import random

from corneto._graph import Graph


class TimeSuite:
    # Define parameters for graph size
    params = ([100, 500], [1000, 5000])  # (num_vertices, num_edges)
    param_names = ["num_vertices", "num_edges"]

    def setup(self, num_vertices, num_edges):
        self.g = Graph()
        self.num_vertices = num_vertices
        self.num_edges = num_edges

        # Add vertices first
        for i in range(self.num_vertices):
            self.g.add_vertex(f"v{i}")

        # Randomly generate edges for the graph
        # Ensure no self-loops for simplicity in random generation
        for _ in range(self.num_edges):
            v1 = random.randint(0, self.num_vertices - 1)
            v2 = random.randint(0, self.num_vertices - 1)
            if v1 == v2:  # Avoid self-loops
                continue
            self.g.add_edge(f"v{v1}", f"v{v2}")

    def time_add_single_vertex(self, num_vertices, num_edges):
        g = Graph()
        for i in range(1000):
            g.add_vertex(f"new_v{i}")

    def time_add_single_edge(self, num_vertices, num_edges):
        g = Graph()
        for i in range(1000):
            g.add_edge(f"a{i}", f"b{i}")

    def time_add_edges(self, num_vertices, num_edges):
        g = Graph()
        edges = [(f"e_a{i}", f"e_b{i}") for i in range(1000)]
        g.add_edges(edges)

    def time_bfs(self, num_vertices, num_edges):
        if self.g.num_vertices > 0:
            self.g.bfs(self.g.V[0])

    def time_bfs_rev(self, num_vertices, num_edges):
        if self.g.num_vertices > 0:
            self.g.bfs(self.g.V[0], reverse=True)

    def time_in_edges(self, num_vertices, num_edges):
        vertices_to_check = self.g.V[: min(100, self.g.num_vertices)]
        list([self.g.in_edges(v) for v in vertices_to_check])

    def time_out_edges(self, num_vertices, num_edges):
        vertices_to_check = self.g.V[: min(100, self.g.num_vertices)]
        list([self.g.out_edges(v) for v in vertices_to_check])

    def time_vertex_incidence_matrix(self, num_vertices, num_edges):
        self.g.vertex_incidence_matrix()

    def time_toposort(self, num_vertices, num_edges):
        # Create a simple DAG for topological sort benchmark
        dag_g = Graph()
        for i in range(num_vertices):
            dag_g.add_vertex(f"v{i}")
        for i in range(num_vertices - 1):
            dag_g.add_edge(f"v{i}", f"v{i + 1}")
        dag_g.toposort()

    def time_copy(self, num_vertices, num_edges):
        self.g.copy()

    def time_subgraph(self, num_vertices, num_edges):
        if self.g.num_vertices > 10:
            subset_vertices = random.sample(self.g.V, min(10, self.g.num_vertices // 2))
            self.g.subgraph(subset_vertices)
        elif self.g.num_vertices > 0:
            self.g.subgraph([self.g.V[0]])

    def time_reachability_analysis(self, num_vertices, num_edges):
        if self.g.num_vertices > 1:
            input_nodes = [self.g.V[0]]
            output_nodes = [self.g.V[-1]]
            self.g.reachability_analysis(input_nodes, output_nodes, verbose=False)
        elif self.g.num_vertices == 1:
            input_nodes = [self.g.V[0]]
            output_nodes = [self.g.V[0]]
            self.g.reachability_analysis(input_nodes, output_nodes, verbose=False)
