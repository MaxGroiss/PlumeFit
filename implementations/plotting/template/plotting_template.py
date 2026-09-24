"""thesis_plot.py - Helfer für konsistente Plots in wissenschaftlichen Dokumenten.

Allgemeine Plot-Vorlage für Arbeiten, Berichte und Paper (Typst oder LaTeX).
Grundidee: Plots werden in der tatsächlichen Textbreite des Dokuments (oder
einem Bruchteil davon) erzeugt und ohne Skalierung eingebunden. So stimmen
die Schriftgrößen im fertigen PDF genau mit denen in thesis.mplstyle überein.

Verwendung:
    import matplotlib.pyplot as plt
    import thesis_plot as tp

    tp.setup(textwidth_mm=160)                   # einmal pro Skript
    fig, ax = plt.subplots(figsize=tp.figsize("wide"))
    ax.plot(t, x, color=tp.C["data"], label="Messsignal")
    ax.set_xlabel(tp.label("t", "s"))            # -> "t in s" (DIN 461)
    ax.set_ylabel(tp.label("U", "V"))
    tp.save(fig, "beispiel")

Einbinden (Breite passend zur gewählten Figurgröße):
    Typst:  #figure(image("plots/beispiel.svg", width: 100%), caption: [...])
    LaTeX:  \\includegraphics[width=\\textwidth]{plots/beispiel.pdf}

Textbreite ermitteln:
    Typst:  Seitenbreite minus linker und rechter Rand (z. B. A4 mit je 25 mm -> 160 mm)
    LaTeX:  \\the\\textwidth im Dokument ausgeben (pt / 2,845 = mm)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import ScalarFormatter

# ---------------------------------------------------------------------------
# Satzspiegel - Voreinstellung, pro Dokument über setup(textwidth_mm=...) ändern
# ---------------------------------------------------------------------------
MM_PER_INCH = 25.4
DEFAULT_TEXTWIDTH_MM = 160.0   # A4 mit 25 mm Innen-/Außenrand
_TEXTWIDTH_MM = DEFAULT_TEXTWIDTH_MM

GOLDEN = (5 ** 0.5 - 1) / 2  # 0,618

# Breite als Anteil der Textbreite, Höhe als Verhältnis zur Breite.
# "half"/"third" lassen etwas Luft für den Abstand zwischen Teilabbildungen.
_KINDS: dict[str, tuple[float, float]] = {
    "single": (1.00, GOLDEN),  # Standard, volle Breite
    "wide":   (1.00, 0.40),    # flach, z. B. Zeitreihen
    "strip":  (1.00, 0.28),    # sehr flach, gestapelte Ausschnitte
    "tall":   (1.00, 1.10),    # mehrere Subplots untereinander
    "square": (0.60, 1.00),
    "half":   (0.49, 0.80),    # zwei nebeneinander
    "third":  (0.32, 0.90),    # drei nebeneinander
}

# ---------------------------------------------------------------------------
# Farben: Paul Tol "bright" (farbenblind-tauglich, auch in Graustufen unterscheidbar)
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

# Semantische Rollen: dieselbe Art von Größe bekommt im ganzen Dokument dieselbe
# Farbe. Für ein konkretes Projekt eigene Rollen ergänzen, z. B.
#     C = {**tp.C, "hintergrund": tp.TOL["orange"]}
C = {
    "data":      TOL["blue"],     # Messdaten / Rohsignal
    "reference": TOL["orange"],   # Referenz, Basislinie, Sollwert
    "derived":   TOL["teal"],     # abgeleitete Größe (Differenz, gefiltertes Signal)
    "model":     TOL["red"],      # Modell, Fit, Simulation
    "secondary": TOL["magenta"],  # zweite Modell-/Vergleichskurve
    "context":   TOL["grey"],     # Hintergrundinformation, Bänder, Einzelkurven
    "marker":    "#777777",       # Ereignismarker (vertikale Linien)
    "highlight": TOL["cyan"],
}

_STYLE_PATH = Path(__file__).resolve().parent / "plotting_style.mplstyle"
_FIGURES_DIR: Path | None = None
_EXTRA_DIRS: list[Path] = []
_DECIMAL_COMMA = True


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup(figures_dir: str | Path | None = None,
          extra_dirs: list[str | Path] | None = None,
          textwidth_mm: float = DEFAULT_TEXTWIDTH_MM,
          decimal_comma: bool = True,
          style: str | Path | None = None,
          font: str = "Libertinus Serif") -> None:
    """Lädt den Style und legt die Dokumentparameter fest. Einmal pro Skript aufrufen.

    :param figures_dir: Zielordner für save(). None -> Ordner "plt_figures"
                        im aktuellen Arbeitsverzeichnis.
    :param extra_dirs: Weitere Ordner, in die save() zusätzlich eine SVG-Kopie
                       schreibt (z. B. direkt in den Bildordner des Dokuments).
    :param textwidth_mm: Textbreite des Zieldokuments in mm.
    :param decimal_comma: Dezimalkomma auf den Achsen (deutschsprachige Dokumente).
    :param style: Anderer Pfad zur .mplstyle-Datei (Default: neben diesem Modul).
    :param font: Schrift, deren Installation geprüft wird (sollte zu font.serif passen).
    """
    global _FIGURES_DIR, _EXTRA_DIRS, _DECIMAL_COMMA, _TEXTWIDTH_MM

    style_path = Path(style) if style else _STYLE_PATH
    if not style_path.exists():
        raise FileNotFoundError(f"Style-Datei nicht gefunden: {style_path}")
    plt.style.use(str(style_path))

    _check_font(font)

    _TEXTWIDTH_MM = float(textwidth_mm)
    _FIGURES_DIR = Path(figures_dir) if figures_dir else Path.cwd() / "plt_figures"
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _EXTRA_DIRS = [Path(d) for d in (extra_dirs or [])]
    for d in _EXTRA_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    _DECIMAL_COMMA = decimal_comma


def _check_font(name: str) -> None:
    """Warnt, wenn die gewünschte Schrift nicht installiert ist (sonst stiller Fallback)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    if name not in names:
        warnings.warn(
            f"Schrift '{name}' nicht gefunden - Plots fallen auf eine Ersatzschrift zurück.\n"
            "Schrift systemweit installieren (Windows: 'Für alle Benutzer installieren') "
            "und danach den matplotlib-Cache löschen:\n"
            f"    {mpl.get_cachedir()}",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Größen und Beschriftung
# ---------------------------------------------------------------------------
def figsize(kind: str = "single", *, width: float | None = None,
            aspect: float | None = None) -> tuple[float, float]:
    """Figurgröße in Zoll, bezogen auf die in setup() gesetzte Textbreite.

    :param kind: single | wide | strip | tall | square | half | third
    :param width: Überschreibt den Breitenanteil (1.0 = volle Textbreite)
    :param aspect: Überschreibt Höhe/Breite
    :return: (Breite, Höhe) in inch
    """
    if kind not in _KINDS:
        raise ValueError(f"Unbekannte Größe '{kind}', erlaubt: {', '.join(_KINDS)}")
    frac, ratio = _KINDS[kind]
    frac = frac if width is None else width
    ratio = ratio if aspect is None else aspect
    w = _TEXTWIDTH_MM / MM_PER_INCH * frac
    return w, w * ratio


def label(symbol: str, unit: str | None = None, *, name: str | None = None,
          math: bool = True) -> str:
    """Achsenbeschriftung nach DIN 461: "Größe in Einheit".

    label("t", "s")                      -> $t$ in s
    label("U", "V", name="Spannung")     -> Spannung $U$ in V
    label(r"\\hat{b}", r"\\mu g\\,m^{-3}")  -> $\\hat{b}$ in µg m⁻³ (Einheit als Mathe)
    label("Anzahl", math=False)          -> Anzahl (ohne Einheit)

    Einheiten mit ^, _ oder \\ werden automatisch aufrecht in Mathe gesetzt.
    """
    sym = f"${symbol}$" if math else symbol
    text = f"{name} {sym}" if name else sym
    if unit:
        if any(c in unit for c in "^_\\"):
            unit = rf"$\mathrm{{{unit}}}$"
        text += f" in {unit}"
    return text


def subplot_labels(axes, labels=None, loc=(0.0, 1.02), **kwargs) -> None:
    """Setzt (a), (b), ... über die Subplots (für Verweise in der Bildunterschrift)."""
    axes = list(axes.flat) if hasattr(axes, "flat") else list(axes)
    labels = labels or [f"({c})" for c in "abcdefghijklmnop"]
    style = dict(fontsize=10, va="bottom", ha="left")
    style.update(kwargs)
    for ax, lbl in zip(axes, labels):
        ax.text(*loc, lbl, transform=ax.transAxes, **style)


def mark_events(ax, positions, **kwargs) -> None:
    """Einheitlich gestaltete vertikale Marker für Ereignisse (Trigger, Zeitpunkte, Grenzen)."""
    style = dict(color=C["marker"], lw=0.6, ls=(0, (2, 2)), zorder=1)
    style.update(kwargs)
    for p in positions:
        ax.axvline(p, **style)


# ---------------------------------------------------------------------------
# Dezimalkomma
# ---------------------------------------------------------------------------
class CommaFormatter(ScalarFormatter):
    """ScalarFormatter mit Dezimalkomma (in Mathe als {,} ohne Zusatzabstand)."""

    def _comma(self, s: str) -> str:
        if "$" in s:
            return re.sub(r"(?<=\d)\.(?=\d)", "{,}", s)
        return re.sub(r"(?<=\d)\.(?=\d)", ",", s)

    def __call__(self, x, pos=None):
        return self._comma(super().__call__(x, pos))

    def get_offset(self):
        return self._comma(super().get_offset())


def apply_decimal_comma(fig) -> None:
    """Ersetzt Standard-Formatter aller Achsen durch den CommaFormatter.

    Log-Achsen, Datums-Achsen und selbst gesetzte Formatter bleiben unberührt.
    """
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if type(fmt) is ScalarFormatter:
                new = CommaFormatter(useMathText=True)
                new.set_powerlimits(mpl.rcParams["axes.formatter.limits"])
                axis.set_major_formatter(new)


# ---------------------------------------------------------------------------
# Speichern
# ---------------------------------------------------------------------------
def save(fig, name: str, formats: tuple[str, ...] = ("svg", "pdf", "png")) -> list[Path]:
    """Speichert die Figur in der festen Größe (kein bbox 'tight').

    svg -> Einbinden in Typst (Text als Pfade, sieht überall gleich aus)
    pdf -> Einbinden in LaTeX bzw. Typst >= 0.14
    png -> schnelle Vorschau, Präsentationen, Notizen

    :param fig: matplotlib Figure
    :param name: Dateiname ohne Endung
    :param formats: Ausgabeformate
    :return: Liste der geschriebenen Pfade
    """
    if _FIGURES_DIR is None:
        raise RuntimeError("tp.setup() muss vor tp.save() aufgerufen werden.")
    if _DECIMAL_COMMA:
        apply_decimal_comma(fig)

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
