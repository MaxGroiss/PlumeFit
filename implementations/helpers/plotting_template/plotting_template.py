"""Helpers for consistent plots in scientific documents (thesis, reports, papers).

General plot template for Typst or LaTeX documents. Core idea: figures are created at
the actual text width of the document (or a fraction of it) and included without
scaling. The font sizes in the final PDF then match plotting_style.mplstyle exactly.

Usage:
    import matplotlib.pyplot as plt
    from implementations.helpers.plotting_template import plotting_template as tp

    tp.setup(textwidth_mm=160)                   # once per script
    fig, ax = plt.subplots(figsize=tp.figsize("wide"))
    ax.plot(t, x, color=tp.C["data"], label="Measured signal")
    ax.set_xlabel(tp.label("t", "s"))            # -> "t in s" (DIN 461)
    ax.set_ylabel(tp.label("U", "V"))
    tp.save(fig, "example")                      # adds the axis arrows

Including the figure (width matching the chosen figure size):
    Typst:  #figure(image("plots/example.svg", width: 100%), caption: [...])
    LaTeX:  \\includegraphics[width=\\textwidth]{plots/example.pdf}

Determining the text width:
    Typst:  page width minus left and right margin (e.g. A4 with 25 mm each -> 160 mm)
    LaTeX:  print \\the\\textwidth in the document (pt / 2.845 = mm)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import font_manager
from matplotlib.patches import ArrowStyle
from matplotlib.ticker import ScalarFormatter
from . import plot_export
# ---------------------------------------------------------------------------
# Type area - default, change per document via setup(textwidth_mm=...)
# ---------------------------------------------------------------------------
MM_PER_INCH = 25.4
DEFAULT_TEXTWIDTH_MM = 160.0   # A4 with 25 mm inner/outer margin
_TEXTWIDTH_MM = DEFAULT_TEXTWIDTH_MM
_EDITABLE = True
GOLDEN = (5 ** 0.5 - 1) / 2  # 0.618

# Width as fraction of the text width, height as ratio to the width.
# "half"/"third" leave some room for the gap between subfigures.
_KINDS: dict[str, tuple[float, float]] = {
    "single": (1.00, GOLDEN),  # default, full width
    "wide":   (1.00, 0.40),    # flat, e.g. time series
    "strip":  (1.00, 0.28),    # very flat, stacked excerpts
    "tall":   (1.00, 1.10),    # several subplots on top of each other
    "square": (0.60, 1.00),
    "half":   (0.49, 0.80),    # two side by side
    "third":  (0.32, 0.90),    # three side by side
}

# ---------------------------------------------------------------------------
# Colors: Paul Tol "bright" (colorblind safe, distinguishable in grayscale)
# ---------------------------------------------------------------------------
TOL = {
    "blue":    "#0077BB",
    "orange":  "#EE7733",
    "teal":    "#009988",
    "red":     "#CC3311",
    "cyan":    "#33BBEE",
    "magenta": "#EE3377",
    "grey":    "#BBBBBB",
    "black":   "#000000",
}

# Semantic roles: the same kind of quantity gets the same color throughout the
# document. Add project specific roles, e.g.
#     C = {**tp.C, "background": tp.TOL["orange"]}
C = {
    "data":      TOL["blue"],     # measured data / raw signal
    "reference": TOL["orange"],   # reference, baseline, set point
    "derived":   TOL["teal"],     # derived quantity (difference, filtered signal)
    "model":     TOL["red"],      # model, fit, simulation
    "secondary": TOL["magenta"],  # second model / comparison curve
    "context":   TOL["grey"],     # background information, bands, individual curves
    "marker":    "#777777",       # event markers (vertical lines)
    "highlight": TOL["cyan"],
}

# Axis arrows: length of the arrow beyond the end of the axis and head geometry
# (relative to ARROW_SCALE_PT, see matplotlib.patches.ArrowStyle)
ARROW_LENGTH_PT = 7.0
ARROW_SCALE_PT = 6.0
ARROW_STYLE = ArrowStyle("-|>", head_length=0.8, head_width=0.35)

UnitStyle = Literal["in", "slash", "brackets"]

_STYLE_PATH = Path(__file__).resolve().parent / "plotting_style.mplstyle"
_FIGURES_DIR: Path | None = None
_EXTRA_DIRS: list[Path] = []
_DECIMAL_COMMA = False
_AXIS_ARROWS = True
_UNIT_STYLE: UnitStyle = "in"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(figures_dir: str | Path | None = None,
          extra_dirs: list[str | Path] | None = None,
          textwidth_mm: float = DEFAULT_TEXTWIDTH_MM,
          decimal_comma: bool = False,
          axis_arrows: bool = True,
          unit_style: UnitStyle = "in",
          style: str | Path | None = None,
          font: str = "Libertinus Serif",
          editable: bool = True) -> None:
    """Load the style and set the document parameters. Call once per script.

    Args:
        figures_dir: Target directory of save(). None -> directory "plt_figures" in the
            current working directory.
        extra_dirs: Further directories in which save() additionally writes an SVG copy
            (e.g. directly into the image folder of the document).
        textwidth_mm: Text width of the target document in mm.
        decimal_comma: Decimal comma on the axes (German documents). Default False.
        axis_arrows: Draw an arrow at the end of every visible axis spine on save().
        unit_style: Unit notation of label(): "in" -> "t in s" (DIN 461),
            "slash" -> "t / s" (ISO 80000-1 quotient), "brackets" -> "t (s)".
        style: Other path to the .mplstyle file (default: next to this module).
        font: Font whose installation is checked (should match font.serif).

    Raises:
        FileNotFoundError: If the style file does not exist.
        ValueError: If unit_style is unknown.
        :param editable:
    """
    global _FIGURES_DIR, _EXTRA_DIRS, _DECIMAL_COMMA, _TEXTWIDTH_MM, _AXIS_ARROWS, _UNIT_STYLE

    style_path = Path(style) if style else _STYLE_PATH
    if not style_path.exists():
        raise FileNotFoundError(f"Style file not found: {style_path}")
    if unit_style not in ("in", "slash", "brackets"):
        raise ValueError(f"Unknown unit_style '{unit_style}', allowed: in, slash, brackets")
    plt.style.use(str(style_path))

    _check_font(font)

    _TEXTWIDTH_MM = float(textwidth_mm)
    _FIGURES_DIR = Path(figures_dir) if figures_dir else Path.cwd() / "plt_figures"
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _EXTRA_DIRS = [Path(d) for d in (extra_dirs or [])]
    for d in _EXTRA_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    _DECIMAL_COMMA = decimal_comma
    _AXIS_ARROWS = axis_arrows
    _UNIT_STYLE = unit_style


def _check_font(name: str) -> None:
    """Warn if the requested font is not installed (otherwise silent fallback)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    if name not in names:
        warnings.warn(
            f"Font '{name}' not found - plots fall back to a substitute font.\n"
            "Install the font system-wide (Windows: 'Install for all users') "
            "and delete the matplotlib cache afterwards:\n"
            f"    {mpl.get_cachedir()}",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Sizes and labels
# ---------------------------------------------------------------------------
def figsize(kind: str = "single", *, width: float | None = None,
            aspect: float | None = None) -> tuple[float, float]:
    """Figure size in inch, relative to the text width set in setup().

    Args:
        kind: single | wide | strip | tall | square | half | third
        width: Overrides the width fraction (1.0 = full text width).
        aspect: Overrides height / width.

    Returns:
        (width, height) in inch.

    Raises:
        ValueError: If kind is unknown.
    """
    if kind not in _KINDS:
        raise ValueError(f"Unknown size '{kind}', allowed: {', '.join(_KINDS)}")
    frac, ratio = _KINDS[kind]
    frac = frac if width is None else width
    ratio = ratio if aspect is None else aspect
    w = _TEXTWIDTH_MM / MM_PER_INCH * frac
    return w, w * ratio


def label(symbol: str, unit: str | None = None, *, name: str | None = None,
          math: bool = True) -> str:
    """Axis label "quantity in unit", notation set by setup(unit_style=...).

    Examples (unit_style="in"):
        label("t", "s")                        -> $t$ in s
        label("U", "V", name="Voltage")        -> Voltage $U$ in V
        label(r"\\hat{b}", r"\\mu g\\,m^{-3}")  -> $\\hat{b}$ in µg m⁻³ (unit set as math)
        label("Count", math=False)             -> Count (without unit)

    Units containing ^, _ or \\ are automatically set upright in math mode.

    Args:
        symbol: Formula symbol (math mode) or plain text if math=False.
        unit: Unit, None for dimensionless quantities.
        name: Optional quantity name in front of the symbol.
        math: Set symbol in math mode.

    Returns:
        Label string for set_xlabel / set_ylabel.
    """
    sym = f"${symbol}$" if math else symbol
    text = f"{name} {sym}" if name else sym
    if unit:
        if any(c in unit for c in "^_\\"):
            unit = rf"$\mathrm{{{unit}}}$"
        if _UNIT_STYLE == "slash":
            text += f" / {unit}"
        elif _UNIT_STYLE == "brackets":
            text += f" ({unit})"
        else:
            text += f" in {unit}"
    return text


def subplot_labels(axes, labels=None, loc=(0.0, 1.02), offset_pt: float | None = None,
                   **kwargs) -> None:
    """Put (a), (b), ... above the subplots (for references in the caption).

    Args:
        axes: Axes array or iterable of Axes.
        labels: Label texts, default (a), (b), ...
        loc: Position in axes coordinates (left edge, bottom of the text).
        offset_pt: Horizontal shift in points. None -> clear of the y-axis arrow
            if axis arrows are active, otherwise 0.
        **kwargs: Passed on to Axes.text.
    """
    axes = list(axes.flat) if hasattr(axes, "flat") else list(axes)
    labels = labels or [f"({c})" for c in "abcdefghijklmnop"]
    if offset_pt is None:
        offset_pt = ARROW_SCALE_PT if _AXIS_ARROWS else 0.0
    style = dict(fontsize=10, va="bottom", ha="left")
    style.update(kwargs)
    for ax, lbl in zip(axes, labels):
        transform = mtransforms.offset_copy(ax.transAxes, fig=ax.figure, units="points", x=offset_pt)
        ax.text(*loc, lbl, transform=transform, **style)


def mark_events(ax, positions, **kwargs) -> None:
    """Uniformly styled vertical markers for events (triggers, points in time, limits)."""
    style = dict(color=C["marker"], lw=0.6, ls=(0, (2, 2)), zorder=1)
    style.update(kwargs)
    for p in positions:
        ax.axvline(p, **style)


# ---------------------------------------------------------------------------
# Axis arrows
# ---------------------------------------------------------------------------
# Spine -> (position of the spine end in axes coordinates, direction of the arrow)
_SPINE_ENDS = {
    "left":   ((0.0, 1.0), (0.0, 1.0)),
    "right":  ((1.0, 1.0), (0.0, 1.0)),
    "bottom": ((1.0, 0.0), (1.0, 0.0)),
    "top":    ((1.0, 1.0), (1.0, 0.0)),
}


def add_axis_arrows(ax, length: float = ARROW_LENGTH_PT) -> None:
    """Extend every visible spine of an axes by an arrow in the direction of increasing values.

    The arrow starts at the end of the spine and reaches `length` points beyond the axes,
    so the data area stays untouched. Line width and color are taken from the spine.
    Calling it twice on the same axes has no effect.

    Assumes spines at the border of the axes (default of plotting_style.mplstyle); spines
    moved with set_position() are not supported.

    Args:
        ax: matplotlib Axes.
        length: Length of the arrow beyond the end of the spine in points.
    """
    if getattr(ax, "_tp_axis_arrows", False):
        return
    for name, (end, direction) in _SPINE_ENDS.items():
        spine = ax.spines.get(name)
        if spine is None or not spine.get_visible():
            continue
        # Tip: end of the spine shifted by `length` points along the axis direction
        tip = mtransforms.offset_copy(ax.transAxes, fig=ax.figure, units="points",
                                      x=direction[0] * length, y=direction[1] * length)
        arrow = ax.annotate("", xy=end, xycoords=tip, xytext=end, textcoords=ax.transAxes,
                            annotation_clip=False, zorder=spine.get_zorder(),
                            arrowprops=dict(arrowstyle=ARROW_STYLE, mutation_scale=ARROW_SCALE_PT,
                                            lw=spine.get_linewidth(), color=spine.get_edgecolor(),
                                            shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))
        # annotate() clips to the axes patch by default -> the arrow would be ignored by
        # get_tightbbox and constrained layout would not reserve space for it
        arrow.set_clip_on(False)
    ax._tp_axis_arrows = True


def apply_axis_arrows(fig) -> None:
    """Add axis arrows to all axes of a figure, colorbars excluded."""
    for ax in fig.get_axes():
        if hasattr(ax, "_colorbar") or not ax.axison:
            continue
        add_axis_arrows(ax)


# ---------------------------------------------------------------------------
# Decimal comma
# ---------------------------------------------------------------------------
class CommaFormatter(ScalarFormatter):
    """ScalarFormatter with decimal comma (in math mode as {,} without extra space)."""

    def _comma(self, s: str) -> str:
        if "$" in s:
            return re.sub(r"(?<=\d)\.(?=\d)", "{,}", s)
        return re.sub(r"(?<=\d)\.(?=\d)", ",", s)

    def __call__(self, x, pos=None):
        return self._comma(super().__call__(x, pos))

    def get_offset(self):
        return self._comma(super().get_offset())


def apply_decimal_comma(fig) -> None:
    """Replace the default formatters of all axes by the CommaFormatter.

    Log axes, date axes and user-defined formatters stay untouched.
    """
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if type(fmt) is ScalarFormatter:
                new = CommaFormatter(useMathText=True)
                new.set_powerlimits(mpl.rcParams["axes.formatter.limits"])
                axis.set_major_formatter(new)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save(fig, name: str, formats: tuple[str, ...] = ("svg", "pdf", "png"), editable: bool = True) -> list[Path]:
    """Save the figure at its fixed size (no bbox 'tight').

    svg -> include in Typst (text as paths, looks the same everywhere)
    pdf -> include in LaTeX or Typst >= 0.14
    png -> quick preview, presentations, notes

    Args:
        fig: matplotlib Figure.
        name: File name without extension.
        formats: Output formats.

    Returns:
        List of the written paths.

    Raises:
        RuntimeError: If setup() was not called before.
        :param editable:
    """
    if _FIGURES_DIR is None:
        raise RuntimeError("tp.setup() must be called before tp.save().")
    # Data + settings for the plotter (before arrows / decimal comma), then the
    # post-processing saved in the plotter (name.plot.json), if there is one
    if _EDITABLE if editable is None else editable:
        outputs = [_FIGURES_DIR / f"{name}.{fmt}" for fmt in formats] + [d / f"{name}.svg" for d in _EXTRA_DIRS]
        plot_export.export_figure(fig, _FIGURES_DIR / name, outputs=outputs, template={
            "file": str(Path(__file__).resolve()), "decimal_comma": _DECIMAL_COMMA, "axis_arrows": _AXIS_ARROWS})
    plot_export.apply_postprocessing(fig, _FIGURES_DIR / f"{name}.plot.json")
    if _DECIMAL_COMMA:
        apply_decimal_comma(fig)
    if _AXIS_ARROWS:
        apply_axis_arrows(fig)

    written = []
    for fmt in formats:
        path = _FIGURES_DIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(path)
    for d in _EXTRA_DIRS:
        path = d / f"{name}.svg"
        fig.savefig(path, format="svg")
        written.append(path)

    w_mm, h_mm = (v * MM_PER_INCH for v in fig.get_size_inches())
    print(f"  {name}: {w_mm:.0f} x {h_mm:.0f} mm -> " + ", ".join(p.name for p in written))
    return written