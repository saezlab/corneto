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

        return super()._create_dummy(name)
