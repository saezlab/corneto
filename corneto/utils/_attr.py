from enum import Enum
from typing import Any, Optional


class Attr(str, Enum):
    """Enum class representing predefined attribute names used in the `Attributes` class.

    Attributes:
        VALUE: Represents the key for a value attribute.
        EDGE_TYPE: Represents the key for an edge type attribute.
        SOURCE_ATTR: Represents the key for a source attribute.
        TARGET_ATTR: Represents the key for a target attribute.
        CUSTOM_ATTR: Represents the key for a custom attribute.
    """

    VALUE = "__value"
    EDGE_TYPE = "__edge_type"
    SOURCE_ATTR = "__source_attr"
    TARGET_ATTR = "__target_attr"
    CUSTOM_ATTR = "__custom_attr"


class Attributes(dict):
    """A dictionary-based class to manage attributes.

    Uses reserved attributes used by CORNETO Graphs.
    """

    __slots__ = ()
    __protected_attrs = set(dir(dict))  # noqa: RUF012

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Set an attribute as a dictionary key-value pair.

        Raises an error if the attribute name is a protected dictionary attribute.

        Args:
            __name (str): The name of the attribute to set.
            __value (Any): The value to assign to the attribute. If the value is an instance
                           of `Attr`, its value is used.

        Raises:
            AttributeError: If trying to set a protected attribute.
        """
        if __name in self.__protected_attrs:
            raise AttributeError(f"'{__name}' is a protected attribute")
        else:
            if isinstance(__value, Attr):
                __value = __value.value
            self.__setitem__(__name, __value)

    def __getattr__(self, __name: str) -> Any:
        """Retrieve the value of an attribute if it exists in the dictionary.

        Args:
            __name (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the requested attribute.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if __name in self:
            return self.__getitem__(__name)
        raise AttributeError(f"{__name} does not exist")

    def set_attr(self, key: Attr, value: Any) -> None:
        """Set an attribute in the dictionary using an `Attr` key.

        Args:
            key (Attr): The key for the attribute, represented as an `Attr` enum.
            value (Any): The value to assign to the key. If the value is an enum,
                         its value is used.
        """
        v = value if not isinstance(value, Enum) else value.value
        self.__setitem__(key.value, v)

    def has_attr(self, key: Attr, value: Optional[Any] = None) -> bool:
        """Check if a given attribute exists in the dictionary, optionally with a specific value.

        Args:
            key (Attr): The key for the attribute, represented as an `Attr` enum.
            value (Optional[Any]): The optional value to check for.

        Returns:
            bool: `True` if the attribute exists and matches the value (if provided),
                  otherwise `False`.
        """
        if value is None:
            return key.value in self
        else:
            v = value if not isinstance(value, Enum) else value.value
            return key.value in self and self[key.value] == v

    def get_attr(self, key: Attr, default: Any = None) -> Any:
        """Retrieve the value of an attribute, or return a default value if not found.

        Args:
            key (Attr): The key for the attribute, represented as an `Attr` enum.
            default (Any): The default value to return if the attribute does not exist.

        Returns:
            Any: The value of the attribute, or the default value if not found.
        """
        if default is not None:
            return self.get(key.value, default)
        return self[key.value]
