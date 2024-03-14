from enum import Enum
from typing import Any, Optional


class Attr(str, Enum):
    VALUE = "__value"
    EDGE_TYPE = "__edge_type"
    SOURCE_ATTR = "__source_attr"
    TARGET_ATTR = "__target_attr"
    CUSTOM_ATTR = "__custom_attr"


class Attributes(dict):
    __slots__ = ()
    __protected_attrs = set(dir(dict))

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in self.__protected_attrs:
            raise AttributeError(f"'{__name}' is a protected attribute")
        else:
            if isinstance(__value, Attr):
                __value = __value.value
            self.__setitem__(__name, __value)

    def __getattr__(self, __name: str):
        if __name in self:
            return self.__getitem__(__name)
        raise AttributeError(f"{__name} does not exist")

    def set_attr(self, key: Attr, value: Any) -> None:
        v = value if not isinstance(value, Enum) else value.value
        self.__setitem__(key.value, v)

    def has_attr(self, key: Attr, value: Optional[Any] = None) -> bool:
        if value is None:
            return key.value in self
        else:
            v = value if not isinstance(value, Enum) else value.value
            return key.value in self and self[key.value] == v

    def get_attr(self, key: Attr, default: Any = None) -> Any:
        if default is not None:
            return self.get(key.value, default)
        return self[key.value]
