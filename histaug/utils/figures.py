from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager

FIGURES_DIR = "/app/figures"

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

DPI = 800


def savefig(name, fig=None):
    if fig is None:
        fig = plt
    fig.savefig(f"{FIGURES_DIR}/{name}.pdf", bbox_inches="tight", dpi=DPI)
    fig.savefig(f"{FIGURES_DIR}/{name}.png", bbox_inches="tight", dpi=DPI)


def rcparams(size="full", w=None, h=None, default_smaller=1, **kwargs):
    from tueplots import axes, bundles, figsizes, fontsizes

    if size == "half":
        rel_width = kwargs.get("rel_width", 1.0)
        kwargs["rel_width"] = rel_width * 0.5

    params = {
        **axes.lines(),
        # **bundles.eccv2024(family="sans-serif"),
        **bundles.eccv2024(),
        **figsizes.eccv2024(**kwargs),
        **fontsizes.eccv2024(default_smaller=default_smaller),
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
