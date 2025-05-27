"""Data handling utilities for CORNETO
======================================

This module provides data handling capabilities to define input data
for CORNETO's methods and algorithms. It provides:

Classes
-------
- :class:`Data`: Main data container that maps sample IDs to Sample objects
- :class:`Sample`: Container for sample features and their associated metadata

Key Features
-----------
- Rich metadata support for data features
- Flexible data import/export methods
- Conversion between different data formats
- Filtering and subsetting capabilities
- Data manipulation and transformation utilities

Examples:
--------
Basic usage with Data and Sample classes:

.. code-block:: python

    >>> from corneto.data import Data, Sample
    >>> # Create a dataset and add samples with features
    >>> dataset = Data()
    >>> dataset.add_sample("patient1", {"age": 45, "treatment": {"value": "drugA", "dose": "high"}})
    >>> print(dataset)
    Dataset(num_samples=1)

    >>> # Convert to dictionary format
    >>> data_dict = dataset.to_dict()
    >>> print(data_dict["patient1"]["treatment"])
    {'value': 'drugA', 'dose': 'high'}

Utilities
---------

The package also provides utility functions for generating random data:

.. code-block:: python

    >>> from corneto.data.util import generate_random_signalling_network
    >>> # Generate a random signaling network
    >>> network = generate_random_signalling_network(n=10, m=3, p_inhibitory=0.3)
    >>> print(f"Generated network with {len(network)} edges")
"""

from ._base import Data, Sample

__all__ = ["Data", "Sample"]
