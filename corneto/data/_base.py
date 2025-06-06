from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union


@dataclass
class Sample:
    """Represents a sample as a collection of features.
    Each feature is stored as a value. If a feature needs metadata,
    it should be represented as a dict with a mandatory "value" key.
    """

    features: Dict[str, Any] = field(default_factory=dict)

    def add_feature(self, name: str, value: Any) -> None:
        """Adds a feature to this sample.
        Raises an error if the feature already exists.
        """
        if name in self.features:
            raise ValueError(f"Feature '{name}' already exists in this sample.")
        self.features[name] = value

    def get_values(self, key: str = "value") -> Dict[str, Any]:
        """Returns a dictionary of feature names and their values."""
        return {name: feat[key] if isinstance(feat, dict) else feat for name, feat in self.features.items()}

    def filter_values(self, predicate: Callable[[str, Any], bool], value_key: str = "value") -> Dict[str, Any]:
        """Filters the features and returns a new dict containing only those that satisfy the predicate."""
        return {
            name: feat[value_key] if isinstance(feat, dict) else feat
            for name, feat in self.features.items()
            if predicate(name, feat)
        }

    def filter_values_by(self, key: str, value: Any, value_key: str = "value") -> Dict[str, Any]:
        """Returns features that have metadata matching the given key and value."""
        return {
            name: feat[value_key]
            for name, feat in self.features.items()
            if isinstance(feat, dict) and feat.get(key) == value
        }

    def get_value(self, feature_name: str, key: str = "value") -> Any:
        """Returns the value of a specific feature."""
        feat = self.features[feature_name]
        return feat[key] if isinstance(feat, dict) else feat

    def filter(self, predicate: Callable[[str, Any], bool]) -> Dict[str, Any]:
        """Filters the features and returns a new dict containing only those that satisfy the predicate."""
        return {name: feat for name, feat in self.features.items() if predicate(name, feat)}

    def filter_by(self, key: str, value: Any) -> Dict[str, Any]:
        """Returns features that have metadata matching the given key and value."""
        return {name: feat for name, feat in self.features.items() if isinstance(feat, dict) and feat.get(key) == value}

    def __getitem__(self, key: str) -> Any:
        return self.features[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.features[key] = value

    def __repr__(self) -> str:
        return f"Sample({self.features!r})"


class Data(UserDict[Any, Sample]):
    """A dataset container that maps sample IDs to Sample objects.

    This class provides methods to add, update, delete, and manipulate samples
    and their associated features. It supports conversions from and to various
    data formats such as dictionaries and tight formats.

    Attributes:
        data (Dict[Any, Sample]): A mapping from sample identifiers to Sample objects.

    Examples:
        Create a new dataset and add a sample with features:

        >>> dataset = Data()
        >>> dataset.add_sample("cell1", {"TP53": -1, "BRCA1": 0.5})
        >>> print(dataset)
        Dataset(num_samples=1)

        Create a dataset from a dictionary:

        >>> raw = {
        ...     "cell1": {"TP53": -1, "BRCA1": 0.5},
        ...     "cell2": {"TP53": 0, "BRCA1": -0.2},
        ... }
        >>> dataset = Data.from_dict(raw)
        >>> print(dataset)
        Dataset(num_samples=2)
    """

    def add_sample(self, sample_id: str, features: Optional[Dict[str, Any]] = None) -> None:
        """Add a new sample with the given features to the dataset.

        Args:
            sample_id (str): Unique identifier for the sample.
            features (Optional[Dict[str, Any]]): A dictionary of feature names and their values.
                Defaults to None.

        Raises:
            ValueError: If a sample with the given ID already exists.

        Examples:
            >>> dataset = Data()
            >>> dataset.add_sample("cell1", {"TP53": -1})
            >>> "cell1" in dataset.data
            True
        """
        if sample_id in self.data:
            raise ValueError(f"Sample with id '{sample_id}' already exists.")
        sample = Sample()
        if features:
            for feat_name, value in features.items():
                sample.add_feature(feat_name, value)
        self.data[sample_id] = sample

    @classmethod
    def from_dict(cls, raw_data: Dict[str, Dict[str, Any]]) -> "Data":
        """Create a Data instance from a nested dictionary.

        Each key in `raw_data` is interpreted as a sample ID, and its value is a dictionary
        mapping feature names to feature data.

        Args:
            raw_data (Dict[str, Dict[str, Any]]): Raw data dictionary.

        Returns:
            Data: A new instance of Data containing the samples and features.

        Examples:
            >>> raw = {
            ...     "cell1": {"TP53": -1, "BRCA1": 0.5},
            ...     "cell2": {"TP53": 0}
            ... }
            >>> dataset = Data.from_dict(raw)
            >>> print(dataset)
            Dataset(num_samples=2)
        """
        dataset = cls()
        for sample_id, features in raw_data.items():
            sample = Sample()
            for feat_name, feat_data in features.items():
                sample.add_feature(feat_name, feat_data)
            dataset.data[sample_id] = sample
        return dataset

    def filter(self, predicate: Callable[[str, str, Any], bool]) -> "Data":
        """Filter features across all samples based on a predicate function.

        Args:
            predicate (Callable[[str, str, Any], bool]): A function that takes a sample ID,
                feature name, and feature value, returning True if the feature should be included.

        Returns:
            Data: A new Data instance containing only the features that satisfy the predicate,
                with empty samples excluded.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": -1, "BRCA1": 0.5},
            ...     "cell2": {"TP53": 0, "EGFR": 1.2},
            ... })
            >>> filtered = dataset.filter(lambda sid, fname, fvalue: isinstance(fvalue, (int, float)) and fvalue > 0)
            >>> filtered.data["cell1"].features
            {'BRCA1': 0.5}
            >>> filtered.data["cell2"].features
            {'EGFR': 1.2}
        """
        new_ds = type(self)()
        for sample_id, sample in self.data.items():
            filtered_features = {}
            for feat_name, feat_value in sample.features.items():
                if predicate(sample_id, feat_name, feat_value):
                    filtered_features[feat_name] = feat_value
            if filtered_features:  # Only add samples that have at least one feature after filtering
                new_sample = Sample()
                for feat_name, feat_value in filtered_features.items():
                    new_sample.features[feat_name] = feat_value
                new_ds.data[sample_id] = new_sample
        return new_ds

    def filter_by(self, key: str, value: Any) -> "Data":
        """Filter features across all samples that have metadata matching the given key and value.

        Args:
            key (str): The metadata key to filter by.
            value (Any): The value that the metadata key should match.

        Returns:
            Data: A new Data instance containing only the features with matching metadata,
                with empty samples excluded.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {
            ...         "TP53": {"value": -1, "type": "tumor_suppressor"},
            ...         "EGFR": {"value": 0.8, "type": "oncogene"}
            ...     },
            ...     "cell2": {
            ...         "KRAS": {"value": 1.2, "type": "oncogene"},
            ...         "BRCA1": {"value": -0.5, "type": "tumor_suppressor"}
            ...     }
            ... })
            >>> filtered = dataset.filter_by("type", "oncogene")
            >>> filtered.data["cell1"].features
            {'EGFR': {'value': 0.8, 'type': 'oncogene'}}
            >>> filtered.data["cell2"].features
            {'KRAS': {'value': 1.2, 'type': 'oncogene'}}
        """
        return self.filter(lambda _, __, feat: isinstance(feat, dict) and feat.get(key) == value)

    def subset_features(self, feature_list: List[str]) -> "Data":
        """Create a new Data instance containing only the specified features.

        Args:
            feature_list (List[str]): List of feature names to retain in the subset.

        Returns:
            Data: A new dataset instance with samples containing only the allowed features.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": -1, "BRCA1": 0.5, "EGFR": 1.2},
            ...     "cell2": {"TP53": 0, "BRCA1": -0.2, "KRAS": 0.8}
            ... })
            >>> subset = dataset.subset_features(["TP53", "KRAS"])
            >>> subset.data["cell1"].features
            {'TP53': -1}
        """
        allowed_features = set(feature_list)
        new_dataset = type(self)()  # Create a new instance of the same Data subclass.
        for sample_id, sample in self.data.items():
            new_sample = Sample()
            for feature_name, feature in sample.features.items():
                if feature_name in allowed_features:
                    new_sample.features[feature_name] = feature
            new_dataset.data[sample_id] = new_sample
        return new_dataset

    def to_tight_format(self) -> Dict[str, List[Any]]:
        """Convert the dataset to a "tight" format, which is a dictionary with lists for samples,
        features, values, and metadata.

        The tight format organizes data in parallel lists where each index corresponds to a
        particular feature of a sample.

        Returns:
            Dict[str, List[Any]]: A dictionary with keys "sample", "feature", "value", and "metadata".

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": {"value": -1, "confidence": 0.9}, "BRCA1": 0.5}
            ... })
            >>> tight = dataset.to_tight_format()
            >>> tight["sample"]
            ['cell1', 'cell1']
        """
        samples, features, values, metadatas = [], [], [], []
        for sample_id, sample in self.data.items():
            for feat_name, feat in sample.features.items():
                samples.append(sample_id)
                features.append(feat_name)
                if isinstance(feat, dict) and "value" in feat:
                    values.append(feat["value"])
                    meta = {k: v for k, v in feat.items() if k != "value"}
                    metadatas.append(meta if meta else None)
                else:
                    values.append(feat)
                    metadatas.append(None)
        return {
            "sample": samples,
            "feature": features,
            "value": values,
            "metadata": metadatas,
        }

    @classmethod
    def from_tight_format(cls, tight_data: Dict[str, List[Any]]) -> "Data":
        """Create a Data instance from a tight format dictionary.

        Args:
            tight_data (Dict[str, List[Any]]): A dictionary with keys "sample", "feature",
                "value", and "metadata" that contains parallel lists.

        Returns:
            Data: A new Data instance constructed from the tight format data.

        Examples:
            >>> tight = {
            ...     "sample": ["cell1", "cell1"],
            ...     "feature": ["TP53", "BRCA1"],
            ...     "value": [-1, 0.5],
            ...     "metadata": [{"confidence": 0.9}, None]
            ... }
            >>> dataset = Data.from_tight_format(tight)
            >>> dataset.data["cell1"].features["TP53"]
            {'value': -1, 'confidence': 0.9}
        """
        dataset = cls()
        for sample_id, feat_name, value, metadata in zip(
            tight_data["sample"],
            tight_data["feature"],
            tight_data["value"],
            tight_data["metadata"],
        ):
            if sample_id not in dataset.data:
                dataset.data[sample_id] = Sample()
            if metadata is not None:
                feat_value = {"value": value, **metadata}
            else:
                feat_value = value
            dataset.data[sample_id].add_feature(feat_name, feat_value)
        return dataset

    def to_sample_value_dict(self, value_key: str = "value") -> Dict[str, Dict[str, dict]]:
        """Convert the dataset to a nested dictionary of sample IDs mapping to feature dictionaries,
        where each feature is represented as a dictionary that includes a value and optional metadata.

        Args:
            value_key (str, optional): The key to use for feature values. Defaults to "value".

        Returns:
            Dict[str, Dict[str, dict]]: A nested dictionary mapping sample IDs to features and their
                associated dictionaries.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": {"value": -1, "confidence": 0.9}, "BRCA1": 0.5}
            ... })
            >>> sv_dict = dataset.to_sample_value_dict()
            >>> sv_dict["cell1"]["TP53"]
            {'value': -1, 'confidence': 0.9}
        """
        result = {}
        for sample_id, sample in self.data.items():
            result[sample_id] = {}
            for feat_name, feat in sample.features.items():
                if isinstance(feat, dict) and value_key in feat:
                    feature_dict = {value_key: feat[value_key]}
                    feature_dict.update({k: v for k, v in feat.items() if k != value_key})
                else:
                    feature_dict = {value_key: feat}
                result[sample_id][feat_name] = feature_dict
        return result

    @classmethod
    def from_sample_value_dict(cls, condition_dict: Dict[str, Dict[str, dict]], value_key: str = "value") -> "Data":
        """Create a Data instance from a nested dictionary of sample IDs to feature dictionaries.

        Each feature dictionary must contain the specified `value_key`.

        Args:
            condition_dict (Dict[str, Dict[str, dict]]): A nested dictionary where each key is a sample ID,
                and each value is a dictionary mapping feature names to feature dictionaries.
            value_key (str, optional): The key that must be present in each feature dictionary.
                Defaults to "value".

        Returns:
            Data: A new Data instance populated with the provided features.

        Raises:
            ValueError: If a feature in a sample is missing the `value_key`.

        Examples:
            >>> condition = {
            ...     "cell1": {
            ...         "TP53": {"value": -1, "confidence": 0.9},
            ...         "BRCA1": {"value": 0.5}
            ...     }
            ... }
            >>> dataset = Data.from_sample_value_dict(condition)
            >>> dataset.data["cell1"].features["TP53"]
            {'value': -1, 'confidence': 0.9}
        """
        dataset = cls()
        for sample_id, features in condition_dict.items():
            sample = Sample()
            for feat_name, feat_data in features.items():
                if value_key not in feat_data:
                    raise ValueError(f"Feature '{feat_name}' in sample '{sample_id}' is missing the '{value_key}' key")
                if len(feat_data) > 1:
                    feat_value = {
                        "value": feat_data[value_key],
                        **{k: v for k, v in feat_data.items() if k != value_key},
                    }
                else:
                    feat_value = feat_data[value_key]
                sample.add_feature(feat_name, feat_value)
            dataset.data[sample_id] = sample
        return dataset

    def delete_sample(self, sample_id: str) -> None:
        """Delete a sample from the dataset by its ID.

        Args:
            sample_id (str): The ID of the sample to delete.

        Raises:
            KeyError: If the sample with the given ID does not exist.

        Examples:
            >>> dataset = Data.from_dict({"cell1": {"TP53": -1}})
            >>> dataset.delete_sample("cell1")
            >>> "cell1" in dataset.data
            False
        """
        if sample_id not in self.data:
            raise KeyError(f"Sample with id '{sample_id}' does not exist.")
        del self.data[sample_id]

    def update_sample(self, sample_id: str, features: Dict[str, Any]) -> None:
        """Update the features of an existing sample or add a new sample if it does not exist.

        Args:
            sample_id (str): The ID of the sample to update.
            features (Dict[str, Any]): A dictionary of features to update or add.

        Examples:
            >>> dataset = Data.from_dict({"cell1": {"TP53": -1}})
            >>> dataset.update_sample("cell1", {"BRCA1": 0.5})
            >>> dataset.data["cell1"].features
            {'TP53': -1, 'BRCA1': 0.5}
        """
        if sample_id not in self.data:
            self.add_sample(sample_id, features)
            return
        sample = self.data[sample_id]
        for feat_name, value in features.items():
            sample.features[feat_name] = value

    def get_feature_across_samples(self, feature_name: str) -> Dict[str, Any]:
        """Retrieve a specific feature from all samples that contain it.

        Args:
            feature_name (str): The name of the feature to retrieve.

        Returns:
            Dict[str, Any]: A dictionary mapping sample IDs to the value of the specified feature.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": -1},
            ...     "cell2": {"TP53": 0, "BRCA1": 0.5}
            ... })
            >>> result = dataset.get_feature_across_samples("TP53")
            >>> result
            {'cell1': -1, 'cell2': 0}
        """
        matching = {}
        for sample_id, sample in self.data.items():
            if feature_name in sample.features:
                matching[sample_id] = sample.features[feature_name]
        return matching

    def __iter__(self):
        """Return an iterator over sample IDs in the dataset.

        Returns:
            Iterator: An iterator over the keys of the internal data dictionary.

        Examples:
            >>> dataset = Data.from_dict({"cell1": {"TP53": -1}})
            >>> for sample_id in dataset:
            ...     print(sample_id)
            cell1
        """
        return iter(self.data)

    def merge(self, other: "Data") -> None:
        """Merge another Data instance into this one.

        For samples that exist in both datasets, update the features with those from the other dataset.
        For new samples, simply add them to the current dataset.

        Args:
            other (Data): Another Data instance to merge into this one.

        Examples:
            >>> ds1 = Data.from_dict({"cell1": {"TP53": -1}})
            >>> ds2 = Data.from_dict({"cell1": {"BRCA1": 0.5}, "cell2": {"TP53": 0}})
            >>> ds1.merge(ds2)
            >>> ds1.data["cell1"].features
            {'TP53': -1, 'BRCA1': 0.5}
        """
        for sample_id, other_sample in other.data.items():
            if sample_id in self.data:
                self.data[sample_id].features.update(other_sample.features)
            else:
                self.data[sample_id] = other_sample

    def filter_samples(self, predicate: Callable[[str, Sample], bool]) -> "Data":
        """Filter samples based on a predicate function.

        Args:
            predicate (Callable[[str, Sample], bool]): A function that takes a sample ID and Sample,
                returning True if the sample should be included.

        Returns:
            Data: A new Data instance containing only the samples that satisfy the predicate.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": -1},
            ...     "cell2": {"TP53": 0}
            ... })
            >>> filtered = dataset.filter_samples(lambda sid, s: s.features.get("TP53", 0) < 0)
            >>> list(filtered.data.keys())
            ['cell1']
        """
        new_ds = type(self)()
        for sample_id, sample in self.data.items():
            if predicate(sample_id, sample):
                new_ds.data[sample_id] = sample
        return new_ds

    def collect_features(
        self,
        metadata_key: str,
        metadata_value: Any,
        *,
        value_key: str = "value",
        by_sample: bool = False,
        return_values: bool = False,
    ) -> Union[Set[Any], Dict[str, Set[Any]]]:
        """Collect feature names or values across samples that have a specific metadata key-value pair.

        This method checks each sample for features where the metadata specified by `metadata_key`
        equals `metadata_value`. It can return a flattened set or a dictionary keyed by sample ID.

        Args:
            metadata_key (str): The metadata key to filter features.
            metadata_value (Any): The metadata value to match.
            value_key (str, optional): The key used to extract feature values. Defaults to "value".
            by_sample (bool, optional): If True, returns a dict mapping sample IDs to sets of features.
                If False, returns a flattened set of features across all samples. Defaults to False.
            return_values (bool, optional): If True, collects the feature values; otherwise collects feature names.
                Defaults to False.

        Returns:
            Union[Set[Any], Dict[str, Set[Any]]]: Either a set of collected features or a dictionary mapping
            sample IDs to sets of features, depending on the `by_sample` flag.

        Examples:
            >>> dataset = Data.from_dict({
            ...     "cell1": {"TP53": {"value": -1, "type": "tumor_suppressor"}},
            ...     "cell2": {"KRAS": {"value": 1, "type": "oncogene"}},
            ...     "cell3": {"BRCA1": {"value": 0.5, "type": "tumor_suppressor"}}
            ... })
            >>> collected = dataset.collect_features("type", "tumor_suppressor", return_values=True)
            >>> isinstance(collected, set)
            True
        """
        results: Dict[str, Set[Any]] = {}

        for sample_id, sample in self.data.items():
            # filter_values_by returns a dict {feature_name: value} for those matching metadata_key == metadata_value
            matching = sample.filter_values_by(metadata_key, metadata_value, value_key=value_key)
            if matching:
                if return_values:
                    # Collect the actual values
                    results[sample_id] = set(matching.values())
                else:
                    # Collect the feature names
                    results[sample_id] = set(matching.keys())

        if by_sample:
            return results
        else:
            # Flatten everything into one set
            merged = set()
            for item_set in results.values():
                merged.update(item_set)
            return merged

    def copy(self) -> "Data":
        """Create a shallow copy of the dataset.

        Each sample's features are shallow-copied; if deep copies are needed, modify accordingly.

        Returns:
            Data: A new instance of Data with copied samples.

        Examples:
            >>> dataset = Data.from_dict({"cell1": {"TP53": -1}})
            >>> new_dataset = dataset.copy()
            >>> new_dataset.data["cell1"].features == dataset.data["cell1"].features
            True
        """
        new_ds = type(self)()
        for sample_id, sample in self.data.items():
            # Shallow-copy of features is assumed sufficient; adjust as needed for deep copies.
            new_ds.data[sample_id] = Sample(features=sample.features.copy())
        return new_ds

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            str: A string showing the number of samples in the dataset.

        Examples:
            >>> dataset = Data.from_dict({"cell1": {"TP53": -1}})
            >>> repr(dataset)
            'Dataset(num_samples=1)'
        """
        return f"Dataset(num_samples={len(self.data)})"
