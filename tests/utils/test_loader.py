import pytest

from histaug.utils.loader import PeekableIterator, slide_loader, patch_loader


def test_peekable_iterator():
    it = PeekableIterator(iter([1, 2, 3]))

    # Test peek without advancing the iterator
    assert it.peek() == 1
    assert next(it) == 1

    # Test advancing the iterator after peek
    assert it.peek() == 2
    assert next(it) == 2

    # Test end of iterator with peek
    assert it.peek() == 3
    assert next(it) == 3

    # Test behavior after end of iterator
    assert it.peek() is None
    with pytest.raises(StopIteration):
        next(it)


def test_patch_loader():
    data = [
        ("patch1", "slide1", "index1"),
        ("patch2", "slide1", "index2"),
        ("patch3", "slide2", "index3"),
    ]
    it = PeekableIterator(iter(data))
    slide1_loader = patch_loader(it, "slide1")
    assert list(slide1_loader) == [
        ("patch1", "slide1", "index1"),
        ("patch2", "slide1", "index2"),
    ]


def test_slide_loader():
    data = [
        ("patch1", "slide1", "index1"),
        ("patch2", "slide1", "index2"),
        ("patch3", "slide2", "index3"),
        ("patch4", "slide3", "index4"),
    ]
    loader = slide_loader(iter(data))

    # Extracting slide1 data
    slide1_name, slide1_patches = next(loader)
    assert slide1_name == "slide1"
    assert list(slide1_patches) == [
        ("patch1", "slide1", "index1"),
        ("patch2", "slide1", "index2"),
    ]

    # Extracting slide2 data
    slide2_name, slide2_patches = next(loader)
    assert slide2_name == "slide2"
    assert list(slide2_patches) == [("patch3", "slide2", "index3")]

    # Extracting slide3 data
    slide3_name, slide3_patches = next(loader)
    assert slide3_name == "slide3"
    assert list(slide3_patches) == [("patch4", "slide3", "index4")]

    # Ensuring no more slides are left
    with pytest.raises(StopIteration):
        next(loader)


def test_slide_loader_partial_consumption():
    data = [
        ("patch1", "slide1", "index1"),
        ("patch2", "slide1", "index2"),
        ("patch3", "slide2", "index3"),
        ("patch4", "slide3", "index4"),
    ]
    loader = slide_loader(iter(data))

    # Extracting slide1 data but only consuming one patch
    slide1_name, slide1_patches = next(loader)
    assert slide1_name == "slide1"
    assert next(slide1_patches) == ("patch1", "slide1", "index1")

    # Extracting slide2 data after partially consuming slide1 patches
    slide2_name, slide2_patches = next(loader)
    assert slide2_name == "slide2"
    assert list(slide2_patches) == [("patch3", "slide2", "index3")]

    # Ensuring subsequent slides and patches are still accessible
    slide3_name, slide3_patches = next(loader)
    assert slide3_name == "slide3"
    assert list(slide3_patches) == [("patch4", "slide3", "index4")]

    # Ensuring no more slides are left
    with pytest.raises(StopIteration):
        next(loader)
