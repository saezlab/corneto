import pytest

from corneto.data._base import Data, Sample


@pytest.fixture
def sample_dataset():
    """Build a dataset with several samples and features.
    Some features have metadata `type="input"`, others `type="output"`,
    and some are plain values without metadata.
    """
    from collections import UserDict

    class Sample:
        def __init__(self, features=None):
            self.features = features or {}

        def filter_values_by(self, key, value, value_key="value"):
            """Returns {feature_name: feature_value} for those matching key==value."""
            return {
                name: feat[value_key] if isinstance(feat, dict) else feat
                for name, feat in self.features.items()
                if (isinstance(feat, dict) and feat.get(key) == value)
            }

    class Dataset(UserDict):
        def __init__(self):
            super().__init__()

        def add_sample(self, sample_id, features=None):
            if sample_id in self.data:
                raise ValueError(f"Sample '{sample_id}' already exists.")
            self.data[sample_id] = Sample(features)

        def collect_features(
            self,
            metadata_key: str,
            metadata_value: any,
            *,
            value_key: str = "value",
            by_sample: bool = False,
            return_values: bool = False,
        ):
            results = {}
            for sample_id, sample in self.data.items():
                matching = sample.filter_values_by(metadata_key, metadata_value, value_key)
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
                merged = set()
                for s in results.values():
                    merged.update(s)
                return merged

    # Build a sample Dataset
    ds = Dataset()
    # Sample 1
    ds.add_sample(
        "s1",
        {
            "f1": {"value": 1, "type": "input"},
            "f2": {"value": 2, "type": "output"},
            "f3": {"value": 3, "type": "input"},
            "f_no_meta": 42,
        },
    )
    # Sample 2
    ds.add_sample(
        "s2",
        {
            "f1": {"value": 11, "type": "input"},
            "f4": {"value": 44, "type": "input"},
            "f2": {"value": 22, "type": "output"},
        },
    )
    # Sample 3
    ds.add_sample(
        "s3",
        {
            "f5": {"value": 55, "type": "output"},
            "f6": {"value": 66, "type": "input"},
        },
    )

    return ds


def test_collect_features_names_union(sample_dataset):
    """Collect feature NAMES (return_values=False),
    across all samples, flattened (by_sample=False),
    for features with 'type' == 'input'.
    """
    collected = sample_dataset.collect_features(
        metadata_key="type",
        metadata_value="input",
        return_values=False,
        by_sample=False,
    )
    # We expect f1, f3, f4, f6 across s1, s2, s3
    assert collected == {"f1", "f3", "f4", "f6"}


def test_collect_features_names_by_sample(sample_dataset):
    """Collect feature NAMES by sample (by_sample=True),
    for features with 'type' == 'output'.
    """
    collected = sample_dataset.collect_features(
        metadata_key="type",
        metadata_value="output",
        return_values=False,
        by_sample=True,
    )
    # f2 in s1 and s2, f5 in s3
    # Expect a dict { "s1": {"f2"}, "s2": {"f2"}, "s3": {"f5"} }
    assert collected == {
        "s1": {"f2"},
        "s2": {"f2"},
        "s3": {"f5"},
    }


def test_collect_features_values_union(sample_dataset):
    """Collect feature VALUES (return_values=True),
    across all samples, flattened,
    for features with 'type' == 'input'.
    """
    collected = sample_dataset.collect_features(
        metadata_key="type", metadata_value="input", return_values=True, by_sample=False
    )
    # We have f1=1, f3=3 in s1; f1=11, f4=44 in s2; f6=66 in s3
    # So union => {1, 3, 11, 44, 66}
    assert collected == {1, 3, 11, 44, 66}


def test_collect_features_values_by_sample(sample_dataset):
    """Collect feature VALUES, grouped by sample,
    for features with 'type' == 'output'.
    """
    collected = sample_dataset.collect_features(
        metadata_key="type", metadata_value="output", return_values=True, by_sample=True
    )
    # s1 => f2=2, s2 => f2=22, s3 => f5=55
    # => {"s1": {2}, "s2": {22}, "s3": {55}}
    assert collected == {
        "s1": {2},
        "s2": {22},
        "s3": {55},
    }


def test_collect_features_no_matches(sample_dataset):
    """If no feature has a certain key/value, expect an empty set or empty dict."""
    # Suppose we search for 'type' == 'nonexistent'
    collected_union = sample_dataset.collect_features(
        metadata_key="type",
        metadata_value="nonexistent",
        return_values=False,
        by_sample=False,
    )
    # Flattened set => should be empty
    assert collected_union == set()

    collected_dict = sample_dataset.collect_features(
        metadata_key="type",
        metadata_value="nonexistent",
        return_values=True,
        by_sample=True,
    )
    # by_sample => should be an empty dict
    assert collected_dict == {}


def test_add_sample():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    assert "s1" in ds.data
    assert ds.data["s1"].features["feat1"] == 10


def test_add_duplicate_sample():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    with pytest.raises(ValueError):
        ds.add_sample("s1", {"feat2": 20})


def test_add_feature():
    sample = Sample()
    sample.add_feature("feat1", 100)
    # Trying to add the same feature should raise an error.
    with pytest.raises(ValueError):
        sample.add_feature("feat1", 200)
    assert sample.features["feat1"] == 100


def test_from_dict():
    raw_data = {
        "s1": {"feat1": 10, "feat2": {"value": 20, "unit": "cm"}},
        "s2": {"feat3": 30},
    }
    ds = Data.from_dict(raw_data)
    assert "s1" in ds.data
    assert ds.data["s1"].features["feat1"] == 10
    assert ds.data["s1"].features["feat2"] == {"value": 20, "unit": "cm"}
    assert ds.data["s2"].features["feat3"] == 30


def test_tight_format_conversion():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10, "feat2": {"value": 20, "unit": "cm"}})
    ds.add_sample("s2", {"feat3": 30})
    tight = ds.to_tight_format()
    ds_new = Data.from_tight_format(tight)

    # Check that samples and features match after conversion round-trip.
    assert set(ds_new.data.keys()) == set(ds.data.keys())
    for sample_id in ds.data:
        for feat in ds.data[sample_id].features:
            assert ds_new.data[sample_id].features[feat] == ds.data[sample_id].features[feat]


def test_sample_value_dict_conversion():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10, "feat2": {"value": 20, "unit": "cm"}})
    svd = ds.to_sample_value_dict()
    ds_new = Data.from_sample_value_dict(svd)

    # Check that the round-trip conversion preserves all sample and feature data.
    assert set(ds_new.data.keys()) == set(ds.data.keys())
    for sample_id in ds.data:
        for feat in ds.data[sample_id].features:
            assert ds_new.data[sample_id].features[feat] == ds.data[sample_id].features[feat]


def test_merge_datasets():
    ds1 = Data()
    ds1.add_sample("s1", {"feat1": 10})
    ds1.add_sample("s2", {"feat2": 20})

    ds2 = Data()
    ds2.add_sample("s1", {"feat1": 100, "feat3": 30})
    ds2.add_sample("s3", {"feat4": 40})

    ds1.merge(ds2)
    # In sample s1, feat1 should be updated and feat3 added.
    assert ds1.data["s1"].features["feat1"] == 100
    assert ds1.data["s1"].features["feat3"] == 30
    # Sample s2 remains unchanged.
    assert ds1.data["s2"].features["feat2"] == 20
    # Sample s3 is added.
    assert "s3" in ds1.data


def test_filter_samples():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    ds.add_sample("s2", {"feat1": 20})
    ds.add_sample("s3", {"feat1": 30})

    # Filter samples where 'feat1' is greater than 15.
    filtered = ds.filter_samples(lambda sid, sample: sample.features.get("feat1", 0) > 15)
    assert set(filtered.data.keys()) == {"s2", "s3"}


def test_get_feature_across_samples():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    ds.add_sample("s2", {"feat1": {"value": 20, "unit": "cm"}})
    ds.add_sample("s3", {"feat2": 30})

    features = ds.get_feature_across_samples("feat1")
    assert "s1" in features
    assert "s2" in features
    assert "s3" not in features


def test_delete_sample():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    ds.delete_sample("s1")
    with pytest.raises(KeyError):
        _ = ds.data["s1"]


def test_update_sample():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    ds.update_sample("s1", {"feat1": 100, "feat2": 20})
    assert ds.data["s1"].features["feat1"] == 100
    assert ds.data["s1"].features["feat2"] == 20
    # Updating a non-existing sample should create it.
    ds.update_sample("s2", {"feat3": 30})
    assert "s2" in ds.data


def test_copy_dataset():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    ds_copy = ds.copy()
    # Modify the original and ensure the copy remains unchanged.
    ds.update_sample("s1", {"feat1": 100})
    assert ds_copy.data["s1"].features["feat1"] == 10


# Additional tests to cover all functions in Dataset and Sample


def test_sample_get_values():
    sample = Sample()
    sample.add_feature("feat1", {"value": 50})
    sample.add_feature("feat2", 20)
    values = sample.get_values()
    assert values["feat1"] == 50
    assert values["feat2"] == 20


def test_sample_filter_values():
    sample = Sample()
    sample.add_feature("feat1", {"value": 50})
    sample.add_feature("feat2", {"value": 30})
    sample.add_feature("feat3", 10)
    filtered = sample.filter_values(lambda name, feat: (feat["value"] if isinstance(feat, dict) else feat) > 20)
    assert filtered["feat1"] == 50
    assert filtered["feat2"] == 30
    assert "feat3" not in filtered


def test_sample_filter_values_by():
    sample = Sample()
    sample.add_feature("feat1", {"value": 50, "unit": "cm"})
    sample.add_feature("feat2", {"value": 30, "unit": "kg"})
    sample.add_feature("feat3", {"value": 10, "unit": "cm"})
    filtered = sample.filter_values_by("unit", "cm")
    assert filtered == {"feat1": 50, "feat3": 10}


def test_sample_get_value():
    sample = Sample()
    sample.add_feature("feat1", {"value": 100})
    sample.add_feature("feat2", 200)
    assert sample.get_value("feat1") == 100
    assert sample.get_value("feat2") == 200


def test_sample_filter():
    sample = Sample()
    sample.add_feature("feat1", 5)
    sample.add_feature("feat2", 15)
    sample.add_feature("feat3", 25)
    filtered = sample.filter(lambda name, f: f > 10)
    assert filtered == {"feat2": 15, "feat3": 25}


def test_sample_filter_by():
    sample = Sample()
    sample.add_feature("feat1", {"value": 5, "tag": "a"})
    sample.add_feature("feat2", {"value": 15, "tag": "b"})
    sample.add_feature("feat3", {"value": 25, "tag": "a"})
    filtered = sample.filter_by("tag", "a")
    assert filtered == {
        "feat1": {"value": 5, "tag": "a"},
        "feat3": {"value": 25, "tag": "a"},
    }


def test_iterator_dataset():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    ds.add_sample("s2", {"feat2": 20})
    keys = [k for k in ds]
    assert set(keys) == {"s1", "s2"}


def test_repr_dataset():
    ds = Data()
    ds.add_sample("s1", {"feat1": 10})
    rep = repr(ds)
    assert "num_samples=1" in rep


def test_subset_features():
    ds = Data()
    # Create sample s1 with three features
    ds.add_sample("s1", {"feat1": 10, "feat2": {"value": 20, "unit": "cm"}, "feat3": 30})
    # Create sample s2 with two features
    ds.add_sample("s2", {"feat2": {"value": 40, "unit": "kg"}, "feat4": 50})

    # Subset the dataset to only include 'feat1' and 'feat2'
    sub_ds = ds.subset_features(["feat1", "feat2"])

    # For sample s1, only feat1 and feat2 should be present
    assert set(sub_ds.data["s1"].features.keys()) == {"feat1", "feat2"}
    # For sample s2, only feat2 should be present
    assert set(sub_ds.data["s2"].features.keys()) == {"feat2"}

    # Also check that the feature values match the originals
    assert sub_ds.data["s1"].features["feat1"] == 10
    assert sub_ds.data["s1"].features["feat2"] == {"value": 20, "unit": "cm"}
    assert sub_ds.data["s2"].features["feat2"] == {"value": 40, "unit": "kg"}


def test_data_filter():
    """Test the filter method on the Data class."""
    ds = Data()
    ds.add_sample("s1", {"feat1": 10, "feat2": 20, "feat3": -5})
    ds.add_sample("s2", {"feat1": 5, "feat2": -10, "feat4": 30})

    # Filter to keep only features with positive values
    filtered = ds.filter(lambda sid, fname, fvalue: isinstance(fvalue, (int, float)) and fvalue > 0)

    assert "s1" in filtered.data
    assert "s2" in filtered.data
    assert set(filtered.data["s1"].features.keys()) == {"feat1", "feat2"}
    assert set(filtered.data["s2"].features.keys()) == {"feat1", "feat4"}
    assert filtered.data["s1"].features["feat1"] == 10
    assert filtered.data["s2"].features["feat4"] == 30

    # Filter with sample ID condition
    filtered_s1 = ds.filter(lambda sid, fname, fvalue: sid == "s1" and fvalue > 0)
    assert "s1" in filtered_s1.data
    assert "s2" not in filtered_s1.data
    assert set(filtered_s1.data["s1"].features.keys()) == {"feat1", "feat2"}

    # Filter with feature name condition
    filtered_feat1 = ds.filter(lambda sid, fname, fvalue: fname == "feat1")
    assert "s1" in filtered_feat1.data
    assert "s2" in filtered_feat1.data
    assert set(filtered_feat1.data["s1"].features.keys()) == {"feat1"}
    assert set(filtered_feat1.data["s2"].features.keys()) == {"feat1"}


def test_data_filter_by():
    """Test the filter_by method on the Data class."""
    ds = Data()
    ds.add_sample(
        "s1",
        {
            "feat1": {"value": 10, "type": "numeric", "importance": "high"},
            "feat2": {"value": 20, "type": "numeric", "importance": "medium"},
            "feat3": {"value": -5, "type": "numeric", "importance": "low"},
        },
    )
    ds.add_sample(
        "s2",
        {
            "feat1": {"value": 5, "type": "numeric", "importance": "medium"},
            "feat2": {"value": -10, "type": "numeric", "importance": "low"},
            "feat4": {"value": 30, "type": "categorical", "importance": "high"},
        },
    )

    # Filter by type
    filtered_numeric = ds.filter_by("type", "numeric")
    assert "s1" in filtered_numeric.data
    assert "s2" in filtered_numeric.data
    assert set(filtered_numeric.data["s1"].features.keys()) == {
        "feat1",
        "feat2",
        "feat3",
    }
    assert set(filtered_numeric.data["s2"].features.keys()) == {"feat1", "feat2"}

    # Filter by importance
    filtered_high = ds.filter_by("importance", "high")
    assert "s1" in filtered_high.data
    assert "s2" in filtered_high.data
    assert set(filtered_high.data["s1"].features.keys()) == {"feat1"}
    assert set(filtered_high.data["s2"].features.keys()) == {"feat4"}

    # Filter that results in empty sample should exclude that sample
    filtered_low = ds.filter_by("importance", "low")
    assert "s1" in filtered_low.data
    assert "s2" in filtered_low.data
    assert set(filtered_low.data["s1"].features.keys()) == {"feat3"}
    assert set(filtered_low.data["s2"].features.keys()) == {"feat2"}


if __name__ == "__main__":
    pytest.main()
