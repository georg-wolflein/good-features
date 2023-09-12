from typing import Any, Iterator, Tuple, Optional


class PeekableIterator:
    """
    An iterator wrapper that provides a peek functionality to look ahead to the
    next item without consuming it.

    Attributes:
        iterator (Iterator): The original iterator to be wrapped.
        buffer (Optional[Any]): A buffer to hold the next item for peeking.
    """

    def __init__(self, iterator: Iterator[Any]):
        """
        Initialize PeekableIterator.

        Args:
            iterator (Iterator): The original iterator to be wrapped.
        """
        self.iterator: Iterator[Any] = iterator
        self.buffer: Optional[Any] = None

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        """
        Return the next item from the iterator. If an item is in the buffer
        (due to peeking), it is returned and removed from the buffer.

        Returns:
            Any: The next item from the iterator.
        """
        if self.buffer:
            item = self.buffer
            self.buffer = None
            return item
        return next(self.iterator)

    def peek(self) -> Optional[Any]:
        """
        Peek at the next item without consuming it. If called multiple times
        consecutively, the same item is returned without advancing the iterator.

        Returns:
            Any: The next item from the iterator, or None if the iterator is exhausted.
        """
        if not self.buffer:
            try:
                self.buffer = next(self.iterator)
            except StopIteration:
                return None
        return self.buffer


def patch_loader(loader: PeekableIterator, current_slide: str) -> Iterator[Tuple[Any, str, Any]]:
    """
    Generate patches from the provided loader for a specific slide.

    Args:
        loader (PeekableIterator): The data loader iterator.
        current_slide (str): The slide identifier.

    Yields:
        Tuple: The next patch data tuple for the current slide.
    """
    while True:
        data = loader.peek()
        if data:
            _, slide, _ = data
            if slide == current_slide:
                yield next(loader)
            else:
                return
        else:
            break


def slide_loader(loader: Iterator[Tuple[Any, str, Any]]) -> Iterator[Tuple[str, Iterator[Tuple[Any, str, Any]]]]:
    """
    Generate slides and their corresponding patches from the provided loader.

    This function creates a generator that yields a tuple containing the slide, and a sub-generator that yields the batches of patches for that slide.

    Args:
        loader (Iterator): The data loader iterator. It should yield tuples of the form (patch, slide, index).

    Yields:
        Tuple[str, Iterator]: A tuple containing the slide identifier and an iterator
        over its patches.
    """
    loader = PeekableIterator(loader)
    last_slide = None
    while True:
        current_data = loader.peek()
        if current_data:
            _, current_slide, _ = current_data

            # If the last slide is the same as the current slide, it means the patches of the last slide
            # were not fully consumed. In such a case, we consume the remaining patches of the last slide.
            if last_slide == current_slide:
                next(loader)
                continue

            last_slide = current_slide
            yield current_slide, patch_loader(loader, current_slide)
        else:
            break
