"""Export a matplotlib figure as data + settings for post-processing in the
Typst-Werkzeuge plotter (VS Code), and re-apply the post-processing later.

Two files are written next to the figure:

    name.plotdata.json   settings: size, layout, subplots, axes, styles of all
                         lines / collections / patches / texts, legend
    name.plotdata.npz    all data points (unchanged, float arrays)

The plotter rebuilds the figure from these files. What you change there
(colors, labels, axis limits, extra lines, dimension arrows ...) is stored in
`name.plot.json` as a small python snippet ("post_code"). The data itself is
never modified.

When the script runs again, `apply_postprocessing(fig, ...)` executes that
snippet on the freshly created figure, so new data and your annotations stay
together.

Usage in a plot template (e.g. plotting_template.save()):

    import plot_export
    plot_export.export_figure(fig, figures_dir / name, template={...})
    plot_export.apply_postprocessing(fig, figures_dir / f"{name}.plot.json")

Only matplotlib and numpy are required.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib import collections as mcoll
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib import text as mtext
from matplotlib.image import AxesImage

FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _hex(c):
    if c is None:
        return None
    try:
        return mcolors.to_hex(c, keep_alpha=True)
    except (ValueError, TypeError):
        return None


def _jsonable(v):
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def _float_array(a):
    """Data as float array; datetimes become matplotlib date numbers."""
    try:
        return np.asarray(a, dtype=float), False
    except (TypeError, ValueError):
        import matplotlib.dates as mdates
        return np.asarray(mdates.date2num(a), dtype=float), True


class _Store:
    """Collects the arrays for the .npz file and hands out keys."""

    def __init__(self):
        self.arrays = {}

    def put(self, key, arr):
        arr = np.asarray(arr)
        if arr.dtype == object:
            arr = arr.astype(float)
        self.arrays[key] = arr
        return key


def _transform_kind(artist, ax):
    # patches combine their own shape transform with the data transform
    t = artist.get_data_transform() if isinstance(artist, mpatches.Patch) else artist.get_transform()
    if t is ax.transData:
        return "data"
    if t is ax.get_xaxis_transform(which="grid"):
        return "xaxis"      # x in data, y in axes fraction (axvline, axvspan)
    if t is ax.get_yaxis_transform(which="grid"):
        return "yaxis"      # x in axes fraction, y in data (axhline, axhspan)
    if t is ax.transAxes:
        return "axes"
    return None


def _dashes(line):
    ls = line.get_linestyle()
    if ls in ("-", "None", "", " ", "none"):
        return ls
    pat = getattr(line, "_unscaled_dash_pattern", None)
    if pat and pat[1]:
        return [float(pat[0] or 0), [float(x) for x in pat[1]]]
    return ls


def _rc_snapshot():
    """rcParams that differ from matplotlib's defaults (style of the template)."""
    out = {}
    default = mpl.rcParamsDefault
    for k, v in mpl.rcParams.items():
        if k.startswith(("backend", "interactive", "webagg", "savefig.directory", "animation", "keymap")):
            continue
        try:
            if v == default.get(k):
                continue
        except Exception:
            pass
        if k == "axes.prop_cycle":
            try:
                out[k] = {"color": [mcolors.to_hex(c) for c in v.by_key().get("color", [])]}
            except Exception:
                pass
            continue
        if _jsonable(v):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(_jsonable(x) for x in v):
            out[k] = list(v)
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _legend_info(leg, ax=None):
    if leg is None:
        return None
    loc = leg._loc
    if isinstance(loc, (tuple, list, np.ndarray)):
        loc = [float(x) for x in loc]
    else:
        codes = {v: k for k, v in mpl.legend.Legend.codes.items()}
        loc = codes.get(loc, "best")
    return {
        "loc": loc,
        "ncols": int(getattr(leg, "_ncols", 1)),
        "frameon": bool(leg.get_frame_on()),
        "title": leg.get_title().get_text(),
        "labels": [t.get_text() for t in leg.get_texts()],
        "visible": bool(leg.get_visible()),
    }


def _export_axes(ax, ai, store, notes):
    info = {
        "xlim": [float(x) for x in ax.get_xlim()],
        "ylim": [float(y) for y in ax.get_ylim()],
        "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
        "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
        "title": ax.get_title(), "title_left": ax.get_title(loc="left"), "title_right": ax.get_title(loc="right"),
        "axison": bool(ax.axison),
        "spines": {k: bool(s.get_visible()) for k, s in ax.spines.items()},
        "grid": {
            "x": bool(ax.xaxis.get_gridlines() and ax.xaxis.get_gridlines()[0].get_visible()),
            "y": bool(ax.yaxis.get_gridlines() and ax.yaxis.get_gridlines()[0].get_visible()),
        },
        "xticklabels_visible": bool(any(t.get_visible() for t in ax.get_xticklabels())) if ax.get_xticklabels() else True,
        "yticklabels_visible": bool(any(t.get_visible() for t in ax.get_yticklabels())) if ax.get_yticklabels() else True,
        "legend": _legend_info(ax.get_legend(), ax),
        "lines": [], "collections": [], "patches": [], "texts": [], "images": [],
    }
    for axis, key in ((ax.xaxis, "x"), (ax.yaxis, "y")):
        loc = axis.get_major_locator()
        if isinstance(loc, mpl.ticker.FixedLocator):
            info[f"{key}ticks"] = [float(v) for v in loc.locs]
            fmt = axis.get_major_formatter()
            if isinstance(fmt, mpl.ticker.FixedFormatter):
                info[f"{key}ticklabels"] = list(fmt.seq)

    # ---- lines (plot, axvline, axhline, ...)
    for li, line in enumerate(ax.lines):
        x, xd = _float_array(line.get_xdata(orig=True))
        y, yd = _float_array(line.get_ydata(orig=True))
        tk = _transform_kind(line, ax)
        if tk is None:
            notes.append(f"Achse {ai}: Linie {li} mit unbekannter Transformation übersprungen")
            info["lines"].append({"skip": True})
            continue
        info["lines"].append({
            "x": store.put(f"a{ai}_l{li}_x", x), "y": store.put(f"a{ai}_l{li}_y", y),
            "xdate": xd, "ydate": yd, "transform": tk,
            "color": _hex(line.get_color()), "ls": _dashes(line), "lw": float(line.get_linewidth()),
            "marker": line.get_marker() if isinstance(line.get_marker(), str) else "o",
            "ms": float(line.get_markersize()), "mfc": _hex(line.get_markerfacecolor()),
            "mec": _hex(line.get_markeredgecolor()), "mew": float(line.get_markeredgewidth()),
            "alpha": line.get_alpha(), "zorder": float(line.get_zorder()),
            "label": line.get_label(), "drawstyle": line.get_drawstyle(), "visible": bool(line.get_visible()),
        })

    # ---- collections (scatter, fill_between, vlines, ...)
    for ci, col in enumerate(ax.collections):
        tk = _transform_kind(col, ax)
        base = {
            "alpha": col.get_alpha(), "zorder": float(col.get_zorder()), "label": col.get_label(),
            "visible": bool(col.get_visible()),
            "lw": [float(v) for v in np.atleast_1d(col.get_linewidth())][:1] or [0.0],
        }
        fc, ec = col.get_facecolor(), col.get_edgecolor()
        def colors(arr, name):
            arr = np.asarray(arr)
            if arr.size == 0:
                return "none"
            if len(arr) == 1:
                return _hex(arr[0])
            return {"array": store.put(f"a{ai}_c{ci}_{name}", arr)}
        if isinstance(col, mcoll.PathCollection) and np.asarray(col.get_offsets()).size:
            off = np.asarray(col.get_offsets(), dtype=float)
            info["collections"].append({**base, "kind": "scatter",
                "offsets": store.put(f"a{ai}_c{ci}_off", off),
                "sizes": store.put(f"a{ai}_c{ci}_s", np.asarray(col.get_sizes(), dtype=float)),
                "fc": colors(fc, "fc"), "ec": colors(ec, "ec"),
                "path": store.put(f"a{ai}_c{ci}_path", col.get_paths()[0].vertices) if col.get_paths() else None,
                "codes": (col.get_paths()[0].codes.tolist() if col.get_paths() and col.get_paths()[0].codes is not None else None),
                "transform": "data"})
        elif isinstance(col, mcoll.PolyCollection) and tk:
            keys = [store.put(f"a{ai}_c{ci}_p{pi}", p.vertices) for pi, p in enumerate(col.get_paths())]
            info["collections"].append({**base, "kind": "polys", "polys": keys, "fc": colors(fc, "fc"),
                                        "ec": colors(ec, "ec"), "transform": tk})
        elif isinstance(col, mcoll.LineCollection) and tk:
            keys = [store.put(f"a{ai}_c{ci}_s{si}", np.asarray(s, dtype=float)) for si, s in enumerate(col.get_segments())]
            info["collections"].append({**base, "kind": "segments", "segments": keys,
                                        "colors": colors(col.get_color(), "col"), "transform": tk,
                                        "ls": "-"})
        else:
            notes.append(f"Achse {ai}: {type(col).__name__} wird nicht unterstützt")
            info["collections"].append({"skip": True})

    # ---- patches (bar, hist, axvspan, ...); equal-styled rectangles are grouped
    groups = []
    for pi, p in enumerate(ax.patches):
        tk = _transform_kind(p, ax)
        if isinstance(p, mpatches.Rectangle) and tk:
            style = (tk, _hex(p.get_facecolor()), _hex(p.get_edgecolor()), float(p.get_linewidth()),
                     p.get_alpha(), float(p.get_zorder()), p.get_hatch(), bool(p.get_visible()))
            rect = (p.get_x(), p.get_y(), p.get_width(), p.get_height())
            label = p.get_label()
            if groups and groups[-1]["kind"] == "rects" and groups[-1]["_style"] == style and not (label and not label.startswith("_")):
                groups[-1]["_rects"].append(rect)
                groups[-1]["end"] = pi + 1
            else:
                groups.append({"kind": "rects", "_style": style, "_rects": [rect], "start": pi, "end": pi + 1,
                               "label": label})
        elif isinstance(p, mpatches.Polygon) and tk:
            groups.append({"kind": "polygon", "xy": store.put(f"a{ai}_p{pi}_xy", np.asarray(p.get_xy(), dtype=float)),
                           "transform": tk, "fc": _hex(p.get_facecolor()), "ec": _hex(p.get_edgecolor()),
                           "lw": float(p.get_linewidth()), "alpha": p.get_alpha(), "zorder": float(p.get_zorder()),
                           "label": p.get_label(), "visible": bool(p.get_visible()), "start": pi, "end": pi + 1})
        else:
            notes.append(f"Achse {ai}: {type(p).__name__} wird nicht unterstützt")
            groups.append({"kind": "skip", "start": pi, "end": pi + 1})
    # labels of bar containers (hist / bar put the legend label on the container)
    cont_labels = {}
    for c in ax.containers:
        lab = c.get_label()
        if lab and not lab.startswith("_") and getattr(c, "patches", None):
            cont_labels[id(c.patches[0])] = lab
    for gi, g in enumerate(groups):
        if g["kind"] == "rects":
            tk, fc, ec, lw, alpha, z, hatch, vis = g.pop("_style")
            r = np.asarray(g.pop("_rects"), dtype=float)
            first = ax.patches[g["start"]]
            lab = cont_labels.get(id(first), g["label"])
            g.update({"rects": store.put(f"a{ai}_g{gi}_r", r), "transform": tk, "fc": fc, "ec": ec, "lw": lw,
                      "alpha": alpha, "zorder": z, "hatch": hatch, "visible": vis, "label": lab})
    info["patches"] = groups

    # ---- texts (ax.text, annotate, subplot labels)
    for ti, t in enumerate(ax.texts):
        entry = {"text": t.get_text(), "color": _hex(t.get_color()), "fontsize": float(t.get_fontsize()),
                 "ha": t.get_ha(), "va": t.get_va(), "rotation": float(t.get_rotation()),
                 "weight": str(t.get_fontweight()), "style": t.get_fontstyle(), "zorder": float(t.get_zorder()),
                 "visible": bool(t.get_visible())}
        if isinstance(t, mtext.Annotation):
            def coords(c):
                return c if isinstance(c, str) else None
            xyc, txc = coords(t.xycoords), coords(t.anncoords)
            if xyc and txc:
                ap = t.arrowprops or None
                if ap:
                    ap = {k: (v if _jsonable(v) else _hex(v) or str(v)) for k, v in ap.items()
                          if k in ("arrowstyle", "color", "lw", "linewidth", "shrinkA", "shrinkB", "connectionstyle", "mutation_scale")}
                entry.update({"kind": "annotation", "xy": [float(v) for v in t.xy],
                              "xytext": [float(v) for v in t.get_position()], "xycoords": xyc, "textcoords": txc,
                              "arrowprops": ap})
                info["texts"].append(entry)
                continue
        tk = _transform_kind(t, ax)
        if tk:
            entry.update({"kind": "text", "x": float(t.get_position()[0]), "y": float(t.get_position()[1]), "transform": tk})
        else:
            # e.g. offset transforms (subplot labels): store the position in axes fraction
            disp = t.get_transform().transform(t.get_position())
            x, y = ax.transAxes.inverted().transform(disp)
            entry.update({"kind": "text", "x": float(x), "y": float(y), "transform": "axes"})
        info["texts"].append(entry)

    # ---- images
    for ii, im in enumerate(ax.images):
        if isinstance(im, AxesImage):
            arr = np.asarray(im.get_array())
            info["images"].append({"array": store.put(f"a{ai}_i{ii}", arr), "extent": [float(v) for v in im.get_extent()],
                                   "cmap": im.get_cmap().name, "vmin": im.norm.vmin, "vmax": im.norm.vmax,
                                   "origin": im.origin, "interpolation": im.get_interpolation(),
                                   "alpha": im.get_alpha(), "zorder": float(im.get_zorder())})
    return info


def export_figure(fig, stem, template=None, outputs=None):
    """Write <stem>.plotdata.json and <stem>.plotdata.npz.

    Args:
        fig: matplotlib Figure (before axis arrows / decimal comma are applied).
        stem: Path without extension, e.g. figures_dir / "beispiel_wide".
        template: Settings of the plot template that the plotter re-applies after
            rebuilding, e.g. {"file": path of the template module, "decimal_comma": False,
            "axis_arrows": True}.
        outputs: Files the template writes for this figure (the plotter regenerates them).

    Returns:
        Path of the .plotdata.json file.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()   # final layout, needed for positions
    store, notes = _Store(), []
    all_axes = fig.get_axes()
    axes = [a for a in all_axes if not hasattr(a, "_colorbar")]
    if len(axes) != len(all_axes):
        notes.append("Colorbars werden nicht übernommen")
    idx = {id(a): i for i, a in enumerate(axes)}

    ax_specs = []
    gridspecs = []
    for ai, ax in enumerate(axes):
        spec = _export_axes(ax, ai, store, notes)
        # geometry: subplot of a gridspec, twin of another axes, or a free position
        ss = ax.get_subplotspec()
        twin = [o for o in ax._twinned_axes.get_siblings(ax) if o is not ax and id(o) in idx]
        geom = {"position": [float(v) for v in ax.get_position().bounds]}
        if twin and idx[id(twin[0])] < ai:
            other = twin[0]
            geom["twin"] = {"of": idx[id(other)], "kind": "x" if ax.get_shared_x_axes().joined(ax, other) else "y"}
        elif ss is not None:
            gs = ss.get_topmost_subplotspec().get_gridspec()
            if gs not in gridspecs:
                gridspecs.append(gs)
            r0, r1 = ss.rowspan.start, ss.rowspan.stop
            c0, c1 = ss.colspan.start, ss.colspan.stop
            geom["grid"] = {"gs": gridspecs.index(gs), "rows": [r0, r1], "cols": [c0, c1]}
        sx = [o for o in ax.get_shared_x_axes().get_siblings(ax) if o is not ax and id(o) in idx and idx[id(o)] < ai]
        sy = [o for o in ax.get_shared_y_axes().get_siblings(ax) if o is not ax and id(o) in idx and idx[id(o)] < ai]
        if sx and "twin" not in geom:
            geom["sharex"] = idx[id(sx[0])]
        if sy and "twin" not in geom:
            geom["sharey"] = idx[id(sy[0])]
        spec["geometry"] = geom
        ax_specs.append(spec)

    gs_specs = []
    for gs in gridspecs:
        nr, nc = gs.get_geometry()
        gs_specs.append({"nrows": nr, "ncols": nc,
                         "width_ratios": [float(v) for v in (gs.get_width_ratios() or [1] * nc)],
                         "height_ratios": [float(v) for v in (gs.get_height_ratios() or [1] * nr)],
                         "wspace": gs.wspace, "hspace": gs.hspace})
    engine = fig.get_layout_engine()
    layout = None
    if engine is not None:
        name = type(engine).__name__
        layout = "constrained" if "Constrained" in name else "tight" if "Tight" in name else None

    leg = fig.legends[0] if fig.legends else None
    spec = {
        "version": FORMAT_VERSION,
        "kind": "plotdata",
        "name": stem.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": str(Path(sys.argv[0]).resolve()) if sys.argv and sys.argv[0] else None,
        "matplotlib": mpl.__version__,
        "rc": _rc_snapshot(),
        "template": template or {},
        "outputs": [str(p) for p in (outputs or [])],
        "figure": {
            "size_in": [float(v) for v in fig.get_size_inches()],
            "dpi": float(fig.dpi),
            "layout": layout,
            "facecolor": _hex(fig.get_facecolor()),
            "suptitle": fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else "",
            "legend": _legend_info(leg),
        },
        "gridspecs": gs_specs,
        "axes": ax_specs,
        "notes": notes,
    }
    # where data and post-processing live (copies of the .json elsewhere point back here)
    spec["home"] = str(stem.with_name(stem.name + ".plotdata.json").resolve())
    data_path = stem.with_name(stem.name + ".plotdata.npz")
    np.savez_compressed(data_path, **store.arrays)
    json_path = stem.with_name(stem.name + ".plotdata.json")
    json_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for n in notes:
        warnings.warn(f"plot_export: {n}", stacklevel=2)
    return json_path


# ---------------------------------------------------------------------------
# Post-processing from the plotter
# ---------------------------------------------------------------------------
def apply_postprocessing(fig, plot_json):
    """Apply the post-processing saved by the plotter (name.plot.json), if it exists.

    Returns True if something was applied.
    """
    p = Path(plot_json)
    if not p.exists():
        return False
    try:
        spec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        warnings.warn(f"plot_export: {p.name} konnte nicht gelesen werden ({e})")
        return False
    code = spec.get("post_code")
    if not code:
        return False
    try:
        exec(compile(code, str(p), "exec"), {"fig": fig, "np": np, "__name__": "post"})
    except Exception as e:  # never break the plot script because of the annotations
        warnings.warn(f"plot_export: Nachbearbeitung aus {p.name} fehlgeschlagen: {type(e).__name__}: {e}")
        return False
    return True
