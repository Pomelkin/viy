from collections.abc import Iterable
from collections.abc import Iterator

from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn


def track[T](
    items: Iterable[T],
    description: str,
    total: int | None = None,
    transient: bool = False,
) -> Iterator[T]:
    """
    Iterate over ``items`` while showing a progress bar in the terminal.

    Outside a terminal the bar is printed once, when the iteration ends.

    Args:
        items (Iterable[T]): Items to iterate over.
        description (str): Text shown next to the bar.
        total (int | None): Number of items; ``None`` takes ``len(items)`` when
            it exists and shows an open-ended bar otherwise.
        transient (bool): Remove the bar once the iteration ends.

    Yields:
        T: Items in order.
    """
    columns = (
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*columns, transient=transient) as progress:
        yield from progress.track(items, total=total, description=description)
    return
