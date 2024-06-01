from tueplots.figsizes import _GOLDEN_RATIO, _PAD_INCHES, _from_base_in, _figsize_to_output_dict
from tueplots.fontsizes import _from_base
from tueplots.fonts import _computer_modern_tex


def _figsizes_pami_half(
    *,
    nrows=1,
    ncols=1,
    constrained_layout=True,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
    pad_inches=_PAD_INCHES,
    rel_width=1.0,
):
    """Double-column (half-width) figures for PAMI."""

    figsize = _from_base_in(
        base_width_in=3.4876,
        rel_width=rel_width,
        height_to_width_ratio=height_to_width_ratio,
        nrows=nrows,
        ncols=ncols,
    )
    return _figsize_to_output_dict(
        figsize=figsize,
        constrained_layout=constrained_layout,
        tight_layout=tight_layout,
        pad_inches=pad_inches,
    )


def _figsizes_pami_full(
    *,
    nrows=1,
    ncols=1,
    constrained_layout=True,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
    pad_inches=_PAD_INCHES,
    rel_width=1.0,
):
    """Double-column (full-width) figures for PAMI."""

    figsize = _from_base_in(
        base_width_in=7.1413,
        rel_width=rel_width,
        height_to_width_ratio=height_to_width_ratio,
        nrows=nrows,
        ncols=ncols,
    )
    return _figsize_to_output_dict(
        figsize=figsize,
        constrained_layout=constrained_layout,
        tight_layout=tight_layout,
        pad_inches=pad_inches,
    )


def _fontsizes_pami(*, default_smaller=1):
    """Font size for PAMI."""
    # PAMI text size is 10, but captions are in size 8.
    # Therefore, we use base 8 instead of 10.
    return _from_base(base=8 + 1 - default_smaller)


def _fonts_pami_tex(*, family="serif"):
    """Fonts for TMLR. LaTeX version."""
    return _computer_modern_tex(family=family)


def _bundles_pami(*, column="half", nrows=1, ncols=1, family="serif"):
    """PAMI bundle."""
    if column == "half":
        size = _figsizes_pami_half(nrows=nrows, ncols=ncols)
    elif column == "full":
        size = _figsizes_pami_full(nrows=nrows, ncols=ncols)
    font_config = _fonts_pami_tex(family=family)
    fontsize_config = _fontsizes_pami()
    return {**font_config, **size, **fontsize_config}
