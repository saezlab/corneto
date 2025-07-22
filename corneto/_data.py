import json
import os
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

from corneto.graph import BaseGraph, Graph

FeatureMapping = Literal["none", "vertex", "edge"]


class Feature:
    """A feature with an identifier, value, mapping type and additional attributes.

    Features are the basic building blocks in corneto. Each feature has a unique
    identifier within its sample, an optional value, and a mapping type that
    determines how it relates to the graph structure.
    """

    __slots__ = ("data",)

    def __init__(
        self,
        *,
        id: Any,
        value: Any = None,
        mapping: FeatureMapping = "none",
        **kwargs: Any,
    ) -> None:
        if id is None:
            raise ValueError("Feature id cannot be None.")
        self.data: Dict[str, Any] = {"id": id, "value": value, "mapping": mapping}
        self.data.update(kwargs)

    @property
    def id(self) -> Any:
        return self.data["id"]

    @property
    def value(self) -> Any:
        return self.data["value"]

    @property
    def mapping(self) -> FeatureMapping:
        return self.data["mapping"]

    def to_dict(self) -> dict:
        return self.data.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "Feature":
        mapping: FeatureMapping = d.get("mapping", "none")
        extra = {k: v for k, v in d.items() if k not in ("id", "value", "mapping")}
        return cls(id=d["id"], value=d.get("value"), mapping=mapping, **extra)

    def __repr__(self) -> str:
        extras = ", ".join(f"{k}={v}" for k, v in self.data.items() if k not in ("id", "value", "mapping"))
        return (
            f"Feature(id={self.id}, value={self.value}, mapping={self.mapping}"
            + (f", {extras}" if extras else "")
            + ")"
        )


class Sample:
    """A collection of unique features.

    Samples represent groups of features that belong together. Features within
    a sample must have unique identifiers.
    """

    def __init__(self, features: Optional[List[Feature]] = None) -> None:
        self._features = features[:] if features else []

    def add(self, feature: Feature) -> None:
        if any(f.id == feature.id for f in self._features):
            raise ValueError(f"Feature with id {feature.id} already exists.")
        self._features.append(feature)

    @property
    def features(self) -> List[Feature]:
        return self._features

    def to_dict(self) -> dict:
        return {"features": [f.to_dict() for f in self._features]}

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(features=[Feature.from_dict(fd) for fd in d.get("features", [])])

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._features)

    def __repr__(self) -> str:
        return f"Sample(n_feats={len(self._features)})"

    # --- Queryable API for Features ---
    @property
    def query(self) -> "SampleQuery":
        return SampleQuery(self._features)


class Data:
    """A container for multiple labeled samples.

    The Data class manages a collection of samples, each identified by a unique key.
    It provides methods for adding, accessing, and querying samples and their features.
    """

    def __init__(self, samples: Optional[Dict[Any, Sample]] = None) -> None:
        self._samples: OrderedDict[Any, Sample] = OrderedDict(samples) if samples else OrderedDict()

    def add_sample(self, key: Any, sample: Sample) -> None:
        self._samples[key] = sample

    @property
    def samples(self) -> Dict[Any, Sample]:
        return self._samples

    def to_dict(self) -> dict:
        return {k: s.to_dict() for k, s in self._samples.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Data":
        samples = {k: Sample.from_dict(s) for k, s in d.items()}
        return cls(samples)

    @classmethod
    def from_cdict(cls, d: dict) -> "Data":
        samples = {}
        for sample_key, features_dict in d.items():
            features = []
            for feat_key, feat_attrs in features_dict.items():
                value = feat_attrs.get("value")
                mapping = feat_attrs.get("mapping", "none")
                extra = {k: v for k, v in feat_attrs.items() if k not in ("value", "mapping")}
                feature = Feature(id=feat_key, value=value, mapping=mapping, **extra)
                features.append(feature)
            samples[sample_key] = Sample(features)
        return cls(samples)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "Data":
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: str, compression: Optional[str] = "xz") -> None:
        if compression == "auto" or compression is None:
            _, ext = os.path.splitext(filepath)
            if ext:
                ext = ext.lower()[1:]
                if ext == "gz":
                    compression = "gzip"
                elif ext == "bz2":
                    compression = "bz2"
                elif ext in ("xz", "lzma"):
                    compression = "xz"
                else:
                    compression = None

        data_str = self.to_json()
        if compression == "gzip":
            import gzip

            with gzip.open(filepath, "wt") as f:
                f.write(data_str)
        elif compression == "bz2":
            import bz2

            with bz2.open(filepath, "wt") as f:
                f.write(data_str)
        elif compression == "xz":
            import lzma

            with lzma.open(filepath, "wt") as f:
                f.write(data_str)
        else:
            with open(filepath, "w") as f:
                f.write(data_str)

    @classmethod
    def load(cls, filepath: str, compression: Optional[str] = "auto") -> "Data":
        if compression == "auto":
            _, ext = os.path.splitext(filepath)
            if ext:
                ext = ext.lower()[1:]
                if ext == "gz":
                    compression = "gzip"
                elif ext == "bz2":
                    compression = "bz2"
                elif ext in ("xz", "lzma"):
                    compression = "xz"
                else:
                    compression = None

        if compression == "gzip":
            import gzip

            with gzip.open(filepath, "rt") as f:
                data_str = f.read()
        elif compression == "bz2":
            import bz2

            with bz2.open(filepath, "rt") as f:
                data_str = f.read()
        elif compression == "xz":
            import lzma

            with lzma.open(filepath, "rt") as f:
                data_str = f.read()
        else:
            with open(filepath, "r") as f:
                data_str = f.read()
        return cls.from_json(data_str)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples.values())

    def __repr__(self) -> str:
        sample_features = [len(sample.features) for sample in self._samples.values()]
        if len(sample_features) > 5:
            features_str = " ".join(map(str, sample_features[:4])) + " ... " + str(sample_features[-1])
        else:
            features_str = " ".join(map(str, sample_features))
        return f"Data(n_samples={len(self._samples)}, n_feats=[{features_str}])"

    def copy(self) -> "Data":
        return Data.from_dict(self.to_dict())

    # --- Queryable API for Samples ---
    @property
    def query(self) -> "DataQuery":
        return DataQuery(self._samples.items())


class SampleQuery:
    """Provides lazy, functional operations on a collection of Features.

    All query operations return a new SampleQuery so that additional
    chaining is possible. Terminal methods like `collect` and `to_list` allow
    extraction of concrete values.
    """

    def __init__(self, features: Union[Iterator[Feature], List[Feature]]) -> None:
        self._features = features

    def filter(self, predicate: Callable[[Feature], bool]) -> "SampleQuery":
        return SampleQuery(f for f in self._features if predicate(f))

    select = filter

    def map(self, func: Callable[[Feature], Feature]) -> "SampleQuery":
        return SampleQuery(func(f) for f in self._features)

    def collect(self) -> Sample:
        return Sample(list(self._features))

    def to_list(self) -> List[Feature]:
        return list(self._features)

    def pluck(self, extractor: Callable[[Feature], Any] = lambda f: f.id) -> set:
        return {extractor(f) for f in self.to_list()}

    def unique(self, keys: Optional[List[str]] = None) -> "SampleQuery":
        """Returns a SampleQuery containing features that are unique with respect to the specified key(s).

        By default, uniqueness is determined by the 'id' attribute. The returned query
        can then be further chained with methods like `to_list` or `pluck`.

        Example:
            # Get a query of features with unique ids
            unique_q = sample.query.unique()
            # Then extract as a list or pluck specific values
            unique_features = unique_q.to_list()
        """
        seen = set()

        def generator():
            for feature in self.to_list():
                key_values = tuple(feature.data.get(k) for k in (keys or ["id"]))
                if key_values not in seen:
                    seen.add(key_values)
                    yield feature

        return SampleQuery(generator())

    def __repr__(self) -> str:
        return f"SampleQuery(n={len(self.to_list())})"


class DataQuery:
    """Provides lazy, functional operations on a collection of (key, sample) pairs.

    In addition to operating on the (key, sample) pairs, methods such as
    filter_features, map_features, and unique allow you to work with the underlying features.
    All operations return a new DataQuery or, when appropriate, a SampleQuery so that
    further chaining is possible.
    """

    def __init__(self, samples: Union[Iterator[Tuple[Any, Sample]], List[Tuple[Any, Sample]]]) -> None:
        self._samples = samples

    def filter(self, predicate: Callable[[Tuple[Any, Sample]], bool]) -> "DataQuery":
        return DataQuery(s for s in self._samples if predicate(s))

    select = filter

    def map(self, func: Callable[[Tuple[Any, Sample]], Tuple[Any, Sample]]) -> "DataQuery":
        return DataQuery(func(s) for s in self._samples)

    def collect(self) -> Data:
        return Data(OrderedDict(list(self._samples)))

    def to_list(self) -> List[Tuple[Any, Sample]]:
        return list(self._samples)

    def filter_features(self, predicate: Callable[[Feature], bool]) -> "DataQuery":
        return self.map(lambda kv: (kv[0], kv[1].query.select(predicate).collect()))

    select_features = filter_features

    def map_features(self, func: Callable[[Feature], Feature]) -> "DataQuery":
        return self.map(lambda kv: (kv[0], kv[1].query.map(func).collect()))

    def pluck_features(self, extractor: Callable[[Feature], Any] = lambda f: f.id) -> set:
        return {extractor(f) for _, sample in self.to_list() for f in sample.features}

    pluck = pluck_features

    def unique(self, keys: Optional[List[str]] = None) -> SampleQuery:
        """Flattens features across all samples and returns a SampleQuery of features
        that are unique with respect to the specified key(s). By default, uniqueness is
        determined by the 'id' attribute.

        This query can then be further chained (e.g. with to_list or pluck) to extract values.

        Example:
            unique_q = data.query.unique()
            unique_features = unique_q.to_list()
        """
        seen = set()

        def generator():
            for _, sample in self.to_list():
                for feature in sample.features:
                    key_values = tuple(feature.data.get(k) for k in (keys or ["id"]))
                    if key_values not in seen:
                        seen.add(key_values)
                        yield feature

        return SampleQuery(generator())

    def __repr__(self) -> str:
        return f"DataQuery(n={len(self.to_list())})"


@dataclass
class GraphData:
    """A container combining a graph structure with associated data.

    GraphData combines a Graph object with a Data object, allowing integrated storage
    and retrieval of graph structures along with their features.

    Attributes:
        graph: A BaseGraph instance representing the graph structure.
        data: A Data instance containing feature data associated with the graph.

    Examples:
        Creating a GraphData object:

        ```python
        import corneto
        import numpy as np

        # Create a graph
        graph = corneto.Graph()
        graph.add_vertices(["A", "B", "C"])
        graph.add_edges([("A", "B"), ("B", "C")])

        # Create data with features
        data = corneto.Data()
        sample = corneto.Sample()
        sample.add(corneto.Feature(id="feat1", value=np.array([1, 2, 3]), mapping="vertex"))
        sample.add(corneto.Feature(id="feat2", value=np.array([4, 5]), mapping="edge"))
        data.add_sample("sample1", sample)

        # Combine into GraphData
        graph_data = corneto.GraphData(graph, data)

        # Save the GraphData to a file
        graph_data.save("example_graph_data.zip")

        # Load the GraphData from a file
        loaded_graph_data = corneto.GraphData.load("example_graph_data.zip")
        ```
    """

    graph: BaseGraph
    data: Data

    def save(self, filepath):
        """Save the GraphData to a zip file.

        Saves both the graph structure and associated data to a single compressed
        file for easy storage and sharing.

        Args:
            filepath: Path where the GraphData should be saved.
        """
        # Add .zip extension if not present
        if not filepath.lower().endswith(".zip"):
            filepath += ".zip"

        # Use a temporary directory to store the files
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.pkl.xz")
            data_path = os.path.join(tmpdir, "data.xz")

            # Save the graph and data into the temporary directory
            self.graph.save(graph_path)
            self.data.save(data_path)

            # Create the zip file and add the two files
            with zipfile.ZipFile(filepath, "w") as zf:
                zf.write(graph_path, "graph.pkl.xz")
                zf.write(data_path, "data.xz")

    @classmethod
    def load(cls, filepath):
        """Load a GraphData object from a zip file.

        Extracts and loads both the graph structure and associated data from a
        previously saved GraphData file.

        Args:
            filepath: Path to the saved GraphData file.

        Returns:
            GraphData: A new GraphData instance containing the loaded graph and data.
        """
        with zipfile.ZipFile(filepath, "r") as zf:
            # Extract the graph and data files to temporary locations
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extract("graph.pkl.xz", tmpdir)
                zf.extract("data.xz", tmpdir)

                graph_path = os.path.join(tmpdir, "graph.pkl.xz")
                data_path = os.path.join(tmpdir, "data.xz")

                # Load the graph and data objects using their existing load methods
                graph = Graph.load(graph_path)
                data = Data.load(data_path)

                return cls(graph, data)
