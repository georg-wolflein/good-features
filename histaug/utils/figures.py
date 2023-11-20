from pathlib import Path
import matplotlib.pyplot as plt
from contextlib import contextmanager

FIGURES_DIR = "/app/figures"

Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


def savefig(name, fig=None):
    if fig is None:
        fig = plt
    fig.savefig(f"{FIGURES_DIR}/{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{FIGURES_DIR}/{name}.png", bbox_inches="tight", dpi=300)


def rcparams(size="full", w=None, h=None, **kwargs):
    from tueplots import axes, bundles, figsizes

    params = {
        **axes.lines(),
        **bundles.cvpr2024(family="sans-serif"),
        **(
            {
                "full": figsizes.cvpr2024_full,
                "half": figsizes.cvpr2024_half,
            }[
                size
            ](**kwargs)
        ),
        "figure.dpi": 300,
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
