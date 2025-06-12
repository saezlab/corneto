"""Dataset module providing core data structures for handling feature-rich datasets.

This module implements the fundamental data structures for managing datasets with
features that can contain both values and associated metadata. It provides:

- Feature: A generic class representing a value with associated metadata
- Dataset: A container class managing collections of features under different conditions

These classes support serialization to/from dictionaries and provide flexible ways
to organize and access the data.

Examples:
    Creating and working with Features:
        >>> feature = Feature(
        ...     value=0.5,
        ...     metadata={"unit": "mM", "type": "concentration"}
        ... )
        >>> feature_dict = feature.to_dict()
        >>> print(feature_dict)
        {'value': 0.5, 'unit': 'mM', 'type': 'concentration'}

    Creating and working with Datasets:
        >>> dataset = Dataset()
        >>> dataset.add_feature(
        ...     "condition1",
        ...     "feature1",
        ...     Feature(value=1.0, metadata={"unit": "mM"})
        ... )
        >>> dataset.add_feature(
        ...     "condition1",
        ...     "feature2",
        ...     Feature(value=2.0, metadata={"unit": "pH"})
        ... )
        >>> data_dict = dataset.to_dict()

"""

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterator, Optional, Tuple, TypeVar

T = TypeVar("T")


def flatten_condition_features(grouped_dict, metadata_key):
    """Converts a nested dictionary of the form:
        {
          condition: {
            meta_value: {
               feature_name: feature_value
            },
            ...
          },
          ...
        }
    into a flattened dictionary of the form:
        {
          condition: {
            feature_name: {
              "value": feature_value,
              <metadata_key>: meta_value
            },
            ...
          },
          ...
        }

    Args:
        grouped_dict (dict): The grouped dictionary.
        metadata_key (str): The name of the metadata field to inject (e.g. "type").

    Returns:
        dict: The flattened dictionary.
    """
    flat_dict = {}

    for condition, meta_groups in grouped_dict.items():
        flat_dict[condition] = {}
        for meta_val, features in meta_groups.items():
            for feature_name, feature_value in features.items():
                flat_dict[condition][feature_name] = {
                    "value": feature_value,
                    metadata_key: meta_val,
                }
    return flat_dict


@dataclass
class Feature(Generic[T]):
    """Represents a feature with a value and associated metadata.

    Attributes:
        value (T): The main value of the feature.
        metadata (Dict[str, Any]): A dictionary containing additional metadata.

    Examples:
        Create a simple feature:
            >>> feature = Feature(value=42)
            >>> print(feature.value)
            42

        Create a feature with metadata:
            >>> feature = Feature(value="high", metadata={"confidence": 0.95})
            >>> print(feature.metadata["confidence"])
            0.95
    """

    value: T
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, feature_dict: Dict[str, Any]) -> "Feature":
        """Creates a Feature instance from a dictionary.

        Expects the dictionary to contain a 'value' key.

        Args:
            feature_dict: A dictionary containing a "value" key and additional
                metadata fields.

        Returns:
            Feature: An instantiated Feature object.

        Raises:
            ValueError: If the 'value' key is missing.

        Examples:
            >>> feature_dict = {"value": 0.5, "unit": "mM", "type": "concentration"}
            >>> feature = Feature.from_dict(feature_dict)
            >>> print(feature.value)
            0.5
            >>> print(feature.metadata)
            {'unit': 'mM', 'type': 'concentration'}
        """
        if "value" not in feature_dict:
            raise ValueError("Missing 'value' in feature dictionary")
        return cls(
            value=feature_dict["value"],
            metadata={k: v for k, v in feature_dict.items() if k != "value"},
        )

    def to_dict(self, return_value_only: bool = False) -> Any:
        """Converts the Feature instance back into a dictionary.

        Args:
            return_value_only (bool): If True, returns only the feature value.

        Returns:
            A dictionary containing the value and metadata, or just the value if
            return_value_only is True.

        Examples:
            Full dictionary conversion:
                >>> feature = Feature(value=0.5, metadata={"unit": "mM"})
                >>> print(feature.to_dict())
                {'value': 0.5, 'unit': 'mM'}

            Value-only conversion:
                >>> print(feature.to_dict(return_value_only=True))
                0.5
        """
        if return_value_only:
            return self.value
        result = {"value": self.value}
        result.update(self.metadata)
        return result


@dataclass
class Dataset:
    """Represents a dataset composed of multiple samples under different conditions.

    Attributes:
        samples: A mapping from condition names to dictionaries mapping feature names
                to Feature instances.

    Examples:
        Create a dataset and add features:
            >>> dataset = Dataset()
            >>> dataset.add_feature(
            ...     "condition1",
            ...     "temperature",
            ...     Feature(value=37.0, metadata={"unit": "celsius"})
            ... )
            >>> dataset.add_feature(
            ...     "condition1",
            ...     "pH",
            ...     Feature(value=7.4, metadata={"precision": 0.1})
            ... )
            >>> print(len(dataset))
            1
    """

    samples: Dict[str, Dict[str, Feature]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, Feature]]]:
        """Allows iteration over the dataset by condition.

        Yields:
            A tuple (condition, features) where features is a dictionary mapping
            feature names to Feature instances.
        """
        for condition, features in self.samples.items():
            yield condition, features

    @classmethod
    def from_dict(cls, data: Dict[str, Dict[str, Dict[str, Any]]], is_flat: bool = False) -> "Dataset":
        """Creates a Dataset instance from a nested dictionary.

        The top-level keys are conditions. If `is_flat` is True, each condition is
        expected to map directly to a dictionary of features (each defined by a
        dictionary with a "value" key and additional metadata). If False, the input
        data is assumed to be grouped by metadata and will be flattened.

        Args:
            data (Dict[str, Dict[str, Dict[str, Any]]]): Nested dictionary representing the dataset.
            is_flat (bool): If True, assumes the input data is in flattened format.
                If False, assumes the data is grouped by metadata and will be flattened.
                Default is False.

        Returns:
            Dataset: An instantiated Dataset object.

        Examples:
            Flat input:
                >>> flat_data = {
                ...     "condition1": {
                ...         "temp": {"value": 37.0, "unit": "celsius"},
                ...         "pH": {"value": 7.4}
                ...     }
                ... }
                >>> dataset = Dataset.from_dict(flat_data, is_flat=True)
                >>> print(dataset.samples["condition1"]["temp"].value)
                37.0

            Grouped input:
                >>> grouped_data = {
                ...     "condition1": {
                ...         "celsius": {"temp": 37.0},
                ...         "neutral": {"pH": 7.4}
                ...     }
                ... }
                >>> dataset = Dataset.from_dict(grouped_data, is_flat=False)
                >>> print(dataset.samples["condition1"]["temp"].value)
                37.0
                >>> print(dataset.samples["condition1"]["temp"].metadata)
                {'type': 'celsius'}
        """
        dataset = cls()
        if not is_flat:
            data = flatten_condition_features(data, metadata_key="type")
        for condition, features in data.items():
            processed_features = {name: Feature.from_dict(feature_data) for name, feature_data in features.items()}
            dataset.samples[condition] = processed_features
        return dataset

    def to_dict(
        self,
        group_by: Optional[str] = None,
        return_value_only: bool = False,
        flatten: bool = False,
        metadata_key: str = "type",
    ) -> Dict[str, Any]:
        """Returns the dataset as a dictionary. Optionally groups features
        by a specified metadata key or flattens the structure.

        Args:
            group_by: Metadata key to group features by. If None, no grouping is done.
            return_value_only: If True, only returns the feature values.
            flatten: If True, flattens the structure. Ignored if group_by is set.
            metadata_key: The metadata key to use when flattening. Default is "type".

        Returns:
            A dictionary representation of the dataset.

        Examples:
            Basic conversion:
                >>> dataset = Dataset()
                >>> dataset.add_feature(
                ...     "condition1",
                ...     "temp",
                ...     Feature(value=37.0, metadata={"type": "physical"})
                ... )
                >>> dataset.add_feature(
                ...     "condition1",
                ...     "pH",
                ...     Feature(value=7.4, metadata={"type": "chemical"})
                ... )
                >>> print(dataset.to_dict())
                {
                    'condition1': {
                        'temp': {'value': 37.0, 'type': 'physical'},
                        'pH': {'value': 7.4, 'type': 'chemical'}
                    }
                }

            Flattened structure:
                >>> print(dataset.to_dict(flatten=True))
                {
                    'condition1': {
                        'temp': {'value': 37.0, 'type': 'physical'},
                        'pH': {'value': 7.4, 'type': 'chemical'}
                    }
                }

            Grouping by metadata:
                >>> print(dataset.to_dict(group_by="type"))
                {
                    'condition1': {
                        'physical': {'temp': {'value': 37.0, 'type': 'physical'}},
                        'chemical': {'pH': {'value': 7.4, 'type': 'chemical'}}
                    }
                }
        """

        def transform_feature(feature: Feature) -> Any:
            return feature.value if return_value_only else feature.to_dict()

        if group_by is not None:
            arranged_samples = {}
            for condition, features in self.samples.items():
                arranged_features = {}
                for feature_name, feature in features.items():
                    meta_value = feature.metadata.get(group_by)
                    if meta_value is not None:
                        if meta_value not in arranged_features:
                            arranged_features[meta_value] = {}
                        arranged_features[meta_value][feature_name] = transform_feature(feature)
                if arranged_features:
                    arranged_samples[condition] = arranged_features
            return arranged_samples

        result = {
            condition: {feature_name: transform_feature(feature) for feature_name, feature in features.items()}
            for condition, features in self.samples.items()
        }

        if flatten and not group_by and not return_value_only:
            return flatten_condition_features(result, metadata_key)

        return result

    def add_feature(self, condition: str, name: str, feature: Feature) -> None:
        """Adds a feature to the dataset under the specified condition.
        If the condition does not exist, it is created.

        Args:
            condition (str): The condition under which to add the feature.
            name (str): The feature's name.
            feature (Feature): The feature instance to add.

        Examples:
            >>> dataset = Dataset()
            >>> feature = Feature(value=42.0, metadata={"unit": "mg/L"})
            >>> dataset.add_feature("treatment_A", "concentration", feature)
            >>> print(dataset.samples["treatment_A"]["concentration"].value)
            42.0
        """
        if condition not in self.samples:
            self.samples[condition] = {}
        self.samples[condition][name] = feature

    def remove_feature(self, condition: str, name: str) -> None:
        """Removes a feature from the dataset under the specified condition.

        Args:
            condition (str): The condition from which to remove the feature.
            name (str): The name of the feature to remove.

        Raises:
            KeyError: If the condition or feature name does not exist.

        Examples:
            >>> dataset = Dataset()
            >>> dataset.add_feature("condition1", "temp",
            ...                    Feature(value=37.0))
            >>> dataset.remove_feature("condition1", "temp")
            >>> print("temp" in dataset.samples["condition1"])
            False
        """
        try:
            del self.samples[condition][name]
            if not self.samples[condition]:
                del self.samples[condition]
        except KeyError:
            raise KeyError(f"Feature '{name}' under condition '{condition}' not found.")
