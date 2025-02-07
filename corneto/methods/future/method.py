from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from corneto import DEFAULT_BACKEND
from corneto._constants import DEFAULT_LB, DEFAULT_UB
from corneto._graph import BaseGraph
from corneto.backend._base import Backend


@dataclass
class Feature:
    """Represents a feature with a value and associated metadata.

    Attributes:
        value (Any): The main value of the feature.
        metadata (Dict[str, Any]): A dictionary containing additional metadata related to the feature. Defaults to an empty dictionary.
    """

    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, feature_dict: Dict[str, Any]) -> 'Feature':
        """Creates a Feature instance from a dictionary.

        Args:
            feature_dict (Dict[str, Any]): A dictionary containing the "value" and other key-value pairs as metadata.

        Returns:
            Feature: An instantiated Feature object with the specified value and metadata.
        """
        return cls(value=feature_dict["value"], metadata={k: v for k, v in feature_dict.items() if k != "value"})

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Feature instance back into a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the value and metadata of the feature.
        """
        result = {"value": self.value}
        result.update(self.metadata)
        return result


@dataclass
class Dataset:
    """Represents a dataset composed of multiple samples under different conditions.

    Attributes:
        samples (Dict[str, Dict[str, Feature]]): A dictionary mapping conditions to another dictionary that maps feature names to their respective Feature instances.
    """

    samples: Dict[str, Dict[str, Feature]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Dict[str, Any]]]) -> 'Dataset':
        """Creates a Dataset instance from a nested dictionary structure.

        Args:
            data (Dict[str, Dict[str, Dict[str, Any]]]): A nested dictionary where each top-level key represents a condition,
                                                         and the corresponding value is another dictionary mapping feature names to their dictionaries.

        Returns:
            Dataset: An instantiated Dataset object with the processed features.
        """
        dataset = cls()
        for condition, features in data.items():
            processed_features = {name: Feature.from_dict(feature_data) for name, feature_data in features.items()}
            dataset.samples[condition] = processed_features
        return dataset

    def to_dict(self, key: str = None, return_value_only: bool = False) -> Dict[str, Any]:
        """Returns the dataset as a dictionary. Optionally arranges by metadata and can return only values.

        Args:
            key (str): Metadata key to arrange features by (e.g., "type"). If not provided, returns samples without specific arrangement.
            return_value_only (bool): If True, returns only the feature values instead of full dictionaries.

        Returns:
            Dict[str, Any]: A dictionary representation of the dataset's samples. Arrangement and content depend on the parameters `key` and `return_value_only`.
        """
        def transform_feature(feature):
            """Helper function to return either just the value or the full dictionary of a feature."""
            return feature.value if return_value_only else feature.to_dict()

        if not key:
            # Return samples with or without metadata based on `return_value_only`
            return {
                condition: {feature_name: transform_feature(feature) for feature_name, feature in features.items()}
                for condition, features in self.samples.items()
            }

        arranged_samples = {}
        for condition, features in self.samples.items():
            arranged_features = {}
            for feature_name, feature in features.items():
                meta_value = feature.metadata.get(key)
                if meta_value is not None:
                    if meta_value not in arranged_features:
                        arranged_features[meta_value] = {}
                    # Include transformed feature
                    arranged_features[meta_value][feature_name] = transform_feature(feature)
            if arranged_features:
                arranged_samples[condition] = arranged_features

        return arranged_samples


class FlowMethod(ABC):
    def __init__(
        self,
        flow_lower_bound: float = DEFAULT_LB,
        flow_upper_bound: float = DEFAULT_UB,
        num_flows: int = 1,
        shared_flow_bounds: bool = False,
        lambd: float = 0.0,
        reg_varname: Optional[str] = None,
        reg_varname_suffix: str = "_OR",
        backend: Optional[Backend] = None,
    ):
        if backend is None:
            backend = DEFAULT_BACKEND
        self._backend = backend
        self._flow_lb = flow_lower_bound
        self._flow_ub = flow_upper_bound
        self._num_flows = num_flows
        self._shared_flow_bounds = shared_flow_bounds
        self._reg_varname = reg_varname
        self._reg_varname_suffix = reg_varname_suffix
        self._lambd = lambd
        self._problem = None
        self._data = None
        self._graph = None

    @abstractmethod
    def preprocess(self, graph: BaseGraph, data: Dataset) -> Tuple[BaseGraph, Any]:
        pass

    @abstractmethod
    def create_flow_based_problem(self, flow_problem, graph: BaseGraph, data: Dataset):
        pass

    def build(self, graph: BaseGraph, data: Dataset):
        self._graph, self._data = self.preprocess(graph, data)
        flow_problem = self.backend.Flow(
            self._graph,
            lb=self._flow_lb,
            ub=self._flow_ub,
            n_flows=self._num_flows,
            shared_bounds=self._shared_flow_bounds,
        )
        # Creates the main problem vars and constraints extending the flow problem
        self._problem = self.create_flow_based_problem(flow_problem, self._graph, self._data)
        # Add structured sparsity regularization
        if self._lambd > 0:
            if self._reg_varname is not None:
                reg_var = self._problem.expr[self._reg_varname]
                newvar_name = self._reg_varname + self._reg_varname_suffix
                self._problem += self.backend.linear_or(reg_var, axis=1, varname=newvar_name)
                self._problem.add_objectives(
                    self._problem.expr[newvar_name].sum(),
                    weights=self._lambd
                )
            else:
                raise ValueError("Parameter lambda > 0 but no regularization variable name provided")
        return self._problem

    @property
    def backend(self):
        return self._backend

