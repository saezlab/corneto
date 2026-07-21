"""Data handling utilities for CORNETO
======================================

This module provides the feature-aware data containers used by CORNETO's
methods and algorithms.

Classes
-------
- :class:`Data`: Main data container that maps sample IDs to samples
- :class:`Sample`: Container for feature objects and their metadata
- :class:`Feature`: A value and its graph mapping metadata
- :class:`GraphData`: A serializable graph and data bundle

Key Features
------------
- Rich metadata support for data features
- Flexible data import/export methods
- Conversion between different data formats
- Filtering and subsetting capabilities
- Data manipulation and transformation utilities

Examples:
---------
Basic usage with Data and Sample classes:

.. code-block:: python

    >>> from corneto.data import Data
    >>> dataset = Data.from_cdict({
    ...     "patient1": {
    ...         "treatment": {"value": "drugA", "mapping": "vertex", "dose": "high"}
    ...     }
    ... })
    >>> print(dataset)
    Data(n_samples=1, n_feats=[1])

    >>> # Convert to dictionary format
    >>> data_dict = dataset.to_dict()
    >>> print(data_dict["patient1"]["features"][0]["value"])
    drugA

Utilities
---------

The package also provides utility functions for generating random data:

.. code-block:: python

    >>> from corneto.data.util import generate_random_signalling_network
    >>> # Generate a random signaling network
    >>> network = generate_random_signalling_network(n=10, m=3, p_inhibitory=0.3)
    >>> print(f"Generated network with {len(network)} edges")
"""

from ._base import Data, Feature, GraphData, Sample

__all__ = ["Data", "Feature", "GraphData", "Sample"]
