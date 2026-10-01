Optimization backend (:mod:`corneto.backend`)
=============================================
.. currentmodule:: corneto.backend

.. automodule:: corneto.backend
    :members:

Total variation
---------------

``Backend.TotalVariation`` builds entrywise L1 graph variation for a nonempty,
real affine CORNETO vector or matrix expression created by the same backend.
Continuous, integer, and binary expressions are supported, with or without
variable bounds. For a vector, the connected axis is its only axis. For a
matrix, ``axis`` selects the connected dimension and defaults to ``-1``;
negative axes follow NumPy conventions. The remaining dimension is summed.

``pairs`` must contain at least one directed ``(source, destination)`` pair
with integer indices in range. Empty input and self-pairs are rejected.
Duplicate and reversed pairs each contribute independently. The registered
signed difference is destination minus source; reversing a pair reverses that
signed value while leaving its magnitude unchanged. Each pair's variation is
the sum of these magnitudes across the other axis. The total variation applies
the pair weights to those sums. The optional one-dimensional ``weights``
sequence or array must contain one finite, nonnegative real value per pair;
scalars are rejected and omitted weights are all one.

Every result registers ``<name>_difference`` and ``<name>_abs_bound`` with
shape ``(R, E)``, ``<name>_variation_by_pair`` with shape ``(E,)``, and scalar
``<name>_total_variation``. Here ``E`` is the number of pairs and ``R`` is the
size of the other axis, or one for a vector. When ``name`` is omitted, a
unique prefix is generated. Explicit prefixes should be distinct in a
composed problem. No objective is added automatically.

The primitive adds continuous variables ``D`` and constraints
``D >= difference`` and ``D >= -difference`` with ``D >= 0``. Thus the
registered total variation is an upper bound until an objective or another
constraint tightens ``D``; ``X`` does not need variable bounds. A positive
coefficient in a minimization objective
makes positively weighted entries tight at an optimum when other uses permit
it; zero weights need not be tight. A budget constraint ``total_variation <=
budget`` still gives the exact projected upper bound on actual variation.
Maximizing total variation or lower-bounding it does not force ``D`` to equal
the absolute difference, so diagnostics should be recomputed from the solved
``difference`` expression. In a maximization objective, subtract a positive
variation penalty. Entrywise L1 graph variation encourages connected entries
to be piecewise constant. Its scale depends on the magnitude and units of
``X``, the number of entries summed across the other axis, and the chosen
weights; multiplying ``X`` by a scalar multiplies the variation by its
absolute value.

For a continuous expression whose rows represent signals over three ordered
conditions, connect adjacent columns and add the variation penalty to any
existing model objective:

.. code-block:: python

    tv = backend.TotalVariation(
        X,
        pairs=[(0, 1), (1, 2)],
        axis=1,
        weights=[1.0, 2.0],
        name="conditions",
    )
    problem += tv
    problem.add_objective(tv.expr.conditions_total_variation, weight=0.5)

The same formulation works for binary expressions over an arbitrary condition
graph. To constrain the total amount of switching, merge the returned problem
and add a budget instead of adding an objective:

.. code-block:: python

    tv = backend.TotalVariation(
        selected,
        pairs=[(0, 1), (0, 2), (2, 3)],
        axis=0,
        name="switches",
    )
    problem += tv
    problem += tv.expr.switches_total_variation <= budget
