import corneto as cn
from corneto.data import Data, Feature, GraphData, Sample


def test_root_data_exports_are_canonical():
    assert cn.Data is Data
    assert cn.Feature is Feature
    assert cn.GraphData is GraphData
    assert cn.Sample is Sample


def test_compact_dict_builds_feature_aware_data():
    data = Data.from_cdict(
        {
            "condition": {
                "A": {"value": 1, "mapping": "vertex", "role": "input"},
                "e0": {"value": 2, "mapping": "edge"},
            }
        }
    )

    features = data.samples["condition"].features
    assert [(f.id, f.value, f.mapping) for f in features] == [
        ("A", 1, "vertex"),
        ("e0", 2, "edge"),
    ]
    assert features[0].data["role"] == "input"
    assert data.query.filter_features(lambda f: f.mapping == "vertex").pluck_features() == {"A"}


def test_data_json_and_compressed_file_roundtrip(tmp_path):
    data = Data.from_cdict({"sample": {"A": {"value": -1, "mapping": "vertex"}}})

    json_roundtrip = Data.from_json(data.to_json())
    assert json_roundtrip.to_dict() == data.to_dict()

    path = tmp_path / "data.xz"
    data.save(str(path))
    assert Data.load(str(path)).to_dict() == data.to_dict()


def test_sample_rejects_duplicate_feature_ids():
    sample = Sample([Feature(id="A")])

    try:
        sample.add(Feature(id="A"))
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("duplicate feature id was accepted")


def test_graph_data_roundtrip(tmp_path):
    graph = cn.Graph()
    graph.add_edge("A", "B", interaction=1)
    data = Data.from_cdict({"sample": {"A": {"value": 1, "mapping": "vertex"}}})

    path = tmp_path / "graph-data"
    GraphData(graph, data).save(str(path))
    loaded = GraphData.load(str(path) + ".zip")

    assert loaded.graph.E == graph.E
    assert loaded.data.to_dict() == data.to_dict()
