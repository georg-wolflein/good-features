from typing import Any, Iterator, Tuple, Optional, Callable, TypeVar, Generic

T = TypeVar("T")


class PeekableIterator(Generic[T]):
    """
    An iterator wrapper that provides a peek functionality to look ahead to the
    next item without consuming it.

    Attributes:
        iterator (Iterator): The original iterator to be wrapped.
        buffer (Optional[T]): A buffer to hold the next item for peeking.
    """

    def __init__(self, iterator: Iterator[T]):
        """
        Initialize PeekableIterator.

        Args:
            iterator (Iterator): The original iterator to be wrapped.
        """
        self.iterator: Iterator[T] = iterator
        self.buffer: Optional[T] = None

    def __len__(self) -> int:
        return len(self.iterator)

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        """
        Return the next item from the iterator. If an item is in the buffer
        (due to peeking), it is returned and removed from the buffer.

        Returns:
            T: The next item from the iterator.
        """
        if self.buffer:
            item = self.buffer
            self.buffer = None
            return item
        return next(self.iterator)

    def peek(self) -> Optional[T]:
        """
        Peek at the next item without consuming it. If called multiple times
        consecutively, the same item is returned without advancing the iterator.

        Returns:
            T: The next item from the iterator, or None if the iterator is exhausted.
        """
        if not self.buffer:
            try:
                self.buffer = next(self.iterator)
            except StopIteration:
                return None
        return self.buffer


class GroupedLoader(Generic[T]):
    def __init__(self, loader: Iterator[T], extract_group_from_item: Callable[[T], Any]):
        """A wrapper around an iterator that groups the data by the group identifier.

        This function creates a generator that yields a tuple containing the group identifier, and a sub-generator that yields the group elements for that group.

        Args:
            loader (Iterator): The iterator to be wrapped.
            extract_group_from_slide (Callable[[T], Any]): A callable that extracts the group identifier from the item.
        """
        self.loader = PeekableIterator(loader)
        self.extract_group_from_item = extract_group_from_item
        self.last_group = None

    def __iter__(self) -> Iterator[Tuple[str, Iterator[T]]]:
        return self

    def __len__(self) -> int:
        return len(self.loader)

    def _group_loader(self, current_group: str) -> Iterator[T]:
        """
        Yield items from the provided loader for a specific group.

        Args:
            loader (PeekableIterator): The data loader iterator.
            current_group (str): The current group identifier.
            extract_group_from_item (Callable[[T], Any]): A callable that extracts the group identifier from the item.

        Yields:
            Tuple: The next patch data tuple for the current slide.
        """
        while True:
            data = self.loader.peek()
            if data:
                group = self.extract_group_from_item(data)
                if group == current_group:
                    yield next(self.loader)
                else:
                    return
            else:
                break

    def __next__(self) -> Tuple[str, Iterator[T]]:
        while (current_data := self.loader.peek()) is not None:
            current_group = self.extract_group_from_item(current_data)

            # If the last slide is the same as the current slide, it means the patches of the last slide
            # were not fully consumed. In such a case, we consume the remaining patches of the last slide.
            if self.last_group == current_group:
                next(self.loader)
                continue

            self.last_group = current_group
            return current_group, self._group_loader(current_group)
        raise StopIteration
