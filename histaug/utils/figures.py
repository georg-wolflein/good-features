from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager

FIGURES_DIR = Path("/app/figures")
JOURNAL_FIGURES_DIR = FIGURES_DIR / "journal"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
JOURNAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 800


def savefig(name, fig=None, journal=False):
    if fig is None:
        fig = plt

    figures_dir = JOURNAL_FIGURES_DIR if journal else FIGURES_DIR

    fig.savefig(f"{figures_dir}/{name}.pdf", bbox_inches="tight", dpi=DPI)
    fig.savefig(f"{figures_dir}/{name}.png", bbox_inches="tight", dpi=DPI)


def rcparams(size="full", w=None, h=None, default_smaller=1, journal: bool = False, **kwargs):
    from tueplots import axes, bundles, figsizes, fontsizes
    from .figures_extra import _figsizes_pami_half, _figsizes_pami_full, _fontsizes_pami, _bundles_pami

    if journal:
        bundle = _bundles_pami
        figsize = _figsizes_pami_half if size == "half" else _figsizes_pami_full
        fontsize = _fontsizes_pami
    else:
        if size == "half":
            rel_width = kwargs.get("rel_width", 1.0)
            kwargs["rel_width"] = rel_width * 0.5

        bundle = bundles.eccv2024
        figsize = figsizes.eccv2024
        fontsize = fontsizes.eccv2024

    params = {
        **axes.lines(),
        # **bundle(family="sans-serif"),
        **bundle(),
        **figsize(**kwargs),
        **fontsize(default_smaller=default_smaller),
        "figure.dpi": DPI,
    }
    if w or h:
        figw, figh = params["figure.figsize"]
        if w:
            figw = w * figw
        if h:
            figh = h * figh
        params["figure.figsize"] = (figw, figh)
    return params


@contextmanager
def rc_context(*args, **kwargs):
    with plt.rc_context(rcparams(*args, **kwargs)):
        yield
