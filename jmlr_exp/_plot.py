# Wong (Okabe-Ito) colorblind-friendly palette.
_WONG_PALETTE = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

# Dash patterns (seaborn / matplotlib on-off sequences).
# Empty string = solid; tuples are (on, off, ...) point lengths.
_DASHES: list = [
    "",  # solid
    (4, 1.5),  # dashed
    (1, 1.5),  # dotted
    (4, 1.5, 1, 1.5),  # dash-dot
    (7, 1.5),  # long dash
    (1, 3),  # sparse dots
]

# Marker styles (one per variant, cycles if more variants than markers).
_MARKERS = ["o", "s", "^", "D", "v", "*", "P"]


def build_style_maps(
    levels: list[str],
) -> tuple[dict[str, object], dict[str, str]]:
    """Return ``(dashes, markers)`` dicts keyed by variant name.

    Pass both to :func:`seaborn.lineplot` via the ``dashes=`` and
    ``markers=`` arguments together with ``style=<col>`` so that overlapping
    lines remain distinguishable by dash pattern and marker shape even when
    their colours are hard to tell apart.
    """
    dashes = {v: _DASHES[i % len(_DASHES)] for i, v in enumerate(levels)}
    markers = {v: _MARKERS[i % len(_MARKERS)] for i, v in enumerate(levels)}
    return dashes, markers
