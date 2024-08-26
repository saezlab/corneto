from abc import ABC, abstractmethod

from corneto import DEFAULT_BACKEND
from corneto._graph import BaseGraph


class CornetoMethod(ABC):
    def __init__(self, backend=DEFAULT_BACKEND):
        self._backend = backend
        self._annotated_flow_graph = None
        self.problem = None

    @abstractmethod
    def create_problem(self):
        raise NotImplementedError

    @abstractmethod
    def map_data(self, graph: BaseGraph, data) -> BaseGraph:
        raise NotImplementedError

    @abstractmethod
    def transform_graph(self, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError
