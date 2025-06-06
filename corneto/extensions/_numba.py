from functools import wraps

import numpy as np

from corneto.extensions._optional import OptionalModule


class OptionalNumba(OptionalModule):
    def __init__(self):
        super().__init__("numba")

    def _create_dummy(self, name):  # type: ignore
        if name in ["uint16", "uint32", "uint64", "int16", "int32", "int64"]:
            return int
        if name in ["float32", "float64", "complex64", "complex128"]:
            return float
        if name == "prange":
            return range
        if name == "guvectorize":
            # Dummy guvectorize decorator for simulation purposes
            def dummy_guvectorize(signature, layout):
                def decorator(func):
                    @wraps(func)
                    def wrapped_func(*args, **kwargs):
                        # Extract input arrays
                        a = args[0]  # The input array `a`
                        window_arr = args[1]  # Scalar `window_arr`

                        # Allocate the output array if not provided
                        if len(args) < 3:
                            out = np.zeros_like(a)
                        else:
                            out = args[2]  # Output array

                        # Iterate over the first axis (rows)
                        for i in range(a.shape[0]):
                            func(a[i], window_arr, out[i])  # Call the function for each row

                        return out

                    return wrapped_func

                return decorator

            return dummy_guvectorize

        return super()._create_dummy(name)
