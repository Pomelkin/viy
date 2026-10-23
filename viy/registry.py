from collections.abc import Iterator


class Registry[T]:
    """Mapping from unique names to registered objects with descriptive lookup errors."""

    def __init__(self, kind: str) -> None:
        """
        Initialize an empty registry.

        Args:
            kind (str): Human readable name of the registered objects, used in error messages.
        """
        self.kind = kind
        self._items: dict[str, T] = {}
        return

    def add(self, name: str, item: T, replace: bool = False) -> None:
        """
        Register ``item`` under ``name``.

        Args:
            name (str): Unique name of the item.
            item (T): Object to register.
            replace (bool): Replace an item registered under the same name, e.g.
                when a notebook cell runs again.

        Raises:
            ValueError: If ``name`` is already taken and ``replace`` is off.
        """
        if name in self._items and not replace:
            raise ValueError(f"{self.kind} '{name}' is already registered.")
        self._items[name] = item
        return

    def get(self, name: str) -> T:
        """
        Return the item registered under ``name``.

        Raises:
            KeyError: If nothing is registered under ``name``.
        """
        if name not in self._items:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}."
            )
        return self._items[name]

    def names(self) -> list[str]:
        """Sorted names of all registered items."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)
