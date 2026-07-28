import csv
import json
import os.path

import matplotlib as mpl

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

# Input format:
# results = [(path_to_metrics_json, method_plot_name), ...]
# Optional extended format:
# results = [(path_to_metrics_json, method_plot_name, hue_group), ...]


MAIN_METRICS = [
    "mae_cf_goal_pred",
    "spearman_goal_pred",
    "pixel_l1_minimality",
    "pair_mean_boundary_concentration_strict",
    "pair_mean_boundary_contrast_strict",
    "pair_mean_mean_distance_to_boundary_strict",
    #"difficulty_weighted_dice",
    "pair_mean_dice_org_cf",
    "order_preserved_wrt_org",
]

APPENDIX_METRICS = [
    # success / minimality
    "mae_cf_goal_pred",
    "spearman_goal_pred",
    "target_success_soft",
    "pixel_l1_minimality",
    # final anatomy metrics
    "pair_mean_boundary_concentration_strict",
    "pair_mean_boundary_contrast_strict",
    "pair_mean_mean_distance_to_boundary_strict",
    "pair_mean_dice_org_cf",
    "difficulty_weighted_dice",
    "order_preserved_wrt_org",
    # supporting strict metrics
    "pair_mean_outside_density_strict",
    "pair_mean_inside_ratio_weighted_strict",
    "pair_mean_roi_precision_strict",
    "pair_mean_boundary_concentration_relaxed",
    "pair_mean_boundary_contrast_relaxed",
    "pair_mean_mean_distance_to_boundary_relaxed",
    # temporal details
    "temporal_mask_l1",
    "temporal_mask_xor_change_ratio",
]

METRIC_META = {
    "cf_validity_mae_ef": {
        "label": "EF target MAE",
        "better": "down",
    },
    "mae_cf_goal_pred": {
        "label": "CF target MAE",
        "better": "down",
    },
    "spearman_goal_pred": {
        "label": "Goal--prediction Spearman",
        "better": "up",
    },
    "target_success_soft": {
        "label": "Target success",
        "better": "up",
    },
    "pixel_l1_minimality": {
        "label": "Pixel $L_1$ minimality",
        "better": "down",
    },
    "pair_mean_boundary_concentration_strict": {
        "label": "Boundary concentration",
        "better": "up",
    },
    "pair_mean_boundary_contrast_strict": {
        "label": "Boundary contrast",
        "better": "up",
    },
    "pair_mean_mean_distance_to_boundary_strict": {
        "label": "Mean dist. to boundary",
        "better": "down",
    },
    "pair_mean_dice_org_cf": {
        "label": "Dice similarity",
        "better": "up",
    },
    "difficulty_weighted_dice": {
        "label": "Difficulty-weighted Dice",
        "better": "up",
    },
    "order_preserved_wrt_org": {
        "label": "Order preserved",
        "better": "up",
    },
    "pair_mean_outside_density_strict": {
        "label": "Outside change density",
        "better": "down",
    },
    "pair_mean_inside_ratio_weighted_strict": {
        "label": "Inside change ratio",
        "better": "up",
    },
    "pair_mean_roi_precision_strict": {
        "label": "ROI precision",
        "better": "up",
    },
    "pair_mean_boundary_concentration_relaxed": {
        "label": "Boundary concentration (relaxed)",
        "better": "up",
    },
    "pair_mean_boundary_contrast_relaxed": {
        "label": "Boundary contrast (relaxed)",
        "better": "up",
    },
    "pair_mean_mean_distance_to_boundary_relaxed": {
        "label": "Mean dist. to boundary (relaxed)",
        "better": "down",
    },
    "temporal_mask_l1": {
        "label": "Temporal mask $L_1$",
        "better": "down",
    },
    "temporal_mask_xor_change_ratio": {
        "label": "Temporal XOR change",
        "better": "down",
    },
}
PARETO_PAIRS = [
    {
        "x": "cf_validity_mae_ef",
        "y": "pixel_l1_minimality",
        "xlabel": "EF target MAE $\\downarrow$",
        "ylabel": "Pixel $L_1$ $\\downarrow$",
        "title": "Validity vs. minimality",
        "good_corner": "upper-right",#"lower-left",
        "invert_x": True,
        "invert_y": True,
    },
    {
        "x": "cf_validity_mae_ef",
        "y": "pair_mean_boundary_concentration_strict",
        "xlabel": "EF target MAE $\\downarrow$",
        "ylabel": "Boundary concentration $\\uparrow$",
        "title": "Validity vs. LV boundary focus",
        "good_corner": "upper-right",#"upper-left",
        "invert_x": True,
        "invert_y": False,
    },
    {
        "x": "pixel_l1_minimality",
        "y": "pair_mean_boundary_concentration_strict",
        "xlabel": "Pixel $L_1$ $\\downarrow$",
        "ylabel": "Boundary concentration $\\uparrow$",
        "title": "Minimality vs. LV boundary focus",
        "good_corner": "upper-right",#"upper-left",
        "invert_x": True,
        "invert_y": False,
    },
]
METHOD_MARKERS = {
    "SPN": "o",
    "MLP": "^",
    "4SCM": "s",
}

PLOT_PAIRS = [
    {
        "x": "cf_validity_mae_ef",
        "y": "pair_mean_dice_org_cf",
        "xlabel": "EF MAE $\\downarrow$",
        "ylabel": "Dice $\\uparrow$",
        "title": "Target success vs. anatomy preservation",
        "good_corner": "upper-left",
    },
    {
        "x": "pair_mean_outside_density_strict",
        "y": "pair_mean_boundary_contrast_strict",
        "xlabel": "Outside change density $\\downarrow$",
        "ylabel": "Boundary contrast $\\uparrow$",
        "title": "Off-target change vs. boundary focus",
        "good_corner": "upper-left",
    },
    {
        "x": "pixel_l1_minimality",
        "y": "cf_validity_mae_ef",
        "xlabel": "Pixel $L_1$ minimality $\\downarrow$",
        "ylabel": "EF MAE $\\downarrow$",
        "title": "Minimality vs. target error",
        "good_corner": "lower-left",
    },
    {
        "x": "order_preserved_wrt_org",
        "y": "cf_validity_spearman_ef",
        "xlabel": "Order preserved $\\uparrow$",
        "ylabel": "EF Spearman $\\uparrow$",
        "title": "Temporal order vs. target trend",
        "good_corner": "upper-right",
    },
]
# =============================================================================
# HELPERS
# =============================================================================
def infer_method_family(method_name: str) -> str:
    name = method_name.lower()
    if "4scm" in name:
        return "4SCM"
    if "spn" in name:
        return "SPN"
    if "mlp" in name:
        return "MLP"
    return method_name



def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def fmt_value(x: Optional[float], decimals: int = 3) -> str:
    if x is None:
        return "--"
    try:
        x = float(x)
        if not np.isfinite(x):
            return "--"
        return f"{x:.{decimals}f}"
    except Exception:
        return "--"


def fmt_mean_std(
    mean: Optional[float],
    std: Optional[float] = None,
    decimals: int = 3,
    show_std: bool = False,
) -> str:
    """Format a table cell as mean or mean ± std.

    The standard deviation is only shown when show_std=True and a finite
    aggregate_std value exists for the respective metric.
    """
    mean_str = fmt_value(mean, decimals=decimals)

    if not show_std or mean_str == "--":
        return mean_str

    std_str = fmt_value(std, decimals=decimals)
    if std_str == "--":
        return mean_str

    return f"{mean_str} $\\pm$ {std_str}"


def arrow_for_metric(metric_name: str) -> str:
    better = METRIC_META.get(metric_name, {}).get("better", None)
    if better == "up":
        return "$\\uparrow$"
    if better == "down":
        return "$\\downarrow$"
    if better == "abs0":
        return "$\\to 0$"
    return ""


def display_name(metric_name: str) -> str:
    base = METRIC_META.get(metric_name, {}).get("label", metric_name)
    arrow = arrow_for_metric(metric_name)
    return f"{base} {arrow}".strip()


def collect_method_results(
    results: Sequence[Union[Tuple[str, str], Tuple[str, str, str]]]
) -> List[Dict[str, Any]]:
    out = []
    for item in results:
        if len(item) == 2:
            path, method_name = item
            hue = method_name
        elif len(item) == 3:
            path, method_name, hue = item
        else:
            raise ValueError("Each result entry must be (path, method_name) or (path, method_name, hue_group).")

        #metrics = load_metrics_json(str(path))

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print('load', path)
                metrics= json.load(f)
            out.append({
                "method": str(method_name),
                "hue": str(hue),
                "path": str(path),
                "metrics": metrics,
                "aggregate_mean": metrics.get("aggregate_mean", {}),
                "aggregate_std": metrics.get("aggregate_std", {}),
            })
    return out
def _method_marker(method_name):
    name = method_name.lower()

    if "4scm" in name:
        return "s"   # square
    if "spn" in name:
        return "o"   # circle
    if "mlp" in name:
        return "^"   # triangle

    return "o"


def _is_4scm(method_name):
    return "4scm" in method_name.lower()


def _param_label(method_result):
    method = method_result["method"]

    if _is_4scm(method):
        return method

    beta = method_result.get("beta", None)
    gamma = method_result.get("gamma", None)

    if beta is None or gamma is None:
        return method

    return f"{method}\n$\\beta={beta:g},\\gamma={gamma:g}$"


def _add_preferred_direction(ax, corner):
    if corner == "lower-left":
        xy = (0.06, 0.08)
        xytext = (0.26, 0.26)
    elif corner == "upper-left":
        xy = (0.06, 0.92)
        xytext = (0.26, 0.74)
    elif corner == "upper-right":
        xy = (0.94, 0.92)
        xytext = (0.72, 0.74)
    elif corner == "upper-right":
        xy = (0.94, 0.92)
        xytext = (0.72, 0.74)
    else:
        return

    ax.annotate(
        "preferred",
        xy=xy,
        xytext=xytext,
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="0.35"),
        fontsize=8,
        color="0.35",
        ha="center",
        va="center",
    )

'''
def plot_pareto_matrix(
        method_results,
        save_dir=None,
        show=True,
        show_std=False,
        gamma_cmap="Blues",
):
    gamma_values = _finite_param_values(method_results, "gamma")
    beta_values = _finite_param_values(method_results, "beta")

    gamma_min = min(gamma_values) if gamma_values else 0.0
    gamma_max = max(gamma_values) if gamma_values else 1.0
    beta_min = min(beta_values) if beta_values else 0.0
    beta_max = max(beta_values) if beta_values else 1.0

    norm = mpl.colors.Normalize(vmin=gamma_min, vmax=gamma_max)
    cmap = plt.get_cmap(gamma_cmap)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    for ax, panel, letter in zip(axes, PARETO_PAIRS, ["A", "B", "C"]):
        for mr in method_results:
            x = get_metric_value(mr, panel["x"])
            y = get_metric_value(mr, panel["y"])

            if x is None or y is None:
                continue

            family = mr["family"]
            marker = METHOD_MARKERS.get(family, "o")

            if family == "4SCM":
                facecolor = "0.72"
                edgecolor = "0.25"
                linewidth = 1.1
            else:
                gamma = mr.get("gamma", None)
                beta = mr.get("beta", None)

                try:
                    gamma = float(gamma)
                except Exception:
                    gamma = gamma_min

                try:
                    beta = float(beta)
                except Exception:
                    beta = beta_min

                facecolor = cmap(norm(gamma))
                edgecolor = "black"
                linewidth = _beta_to_linewidth(beta, beta_min, beta_max)

            if show_std:
                sx = get_metric_std(mr, panel["x"])
                sy = get_metric_std(mr, panel["y"])

                ax.errorbar(
                    x,
                    y,
                    xerr=sx,
                    yerr=sy,
                    fmt="none",
                    ecolor="0.45",
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.55,
                    zorder=1,
                )

            ax.scatter(
                x,
                y,
                s=80,
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.95,
                zorder=3,
            )

        _add_preferred_direction(ax, panel["good_corner"])

        ax.set_title(f"({letter}) {panel['title']}", fontsize=10)
        ax.set_xlabel(panel["xlabel"], fontsize=9)
        ax.set_ylabel(panel["ylabel"], fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8)

    # Method legend
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS["SPN"],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=7,
            label="SPN",
        ),
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS["MLP"],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=7,
            label="MLP",
        ),
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS["4SCM"],
            linestyle="",
            markerfacecolor="0.72",
            markeredgecolor="0.25",
            markeredgewidth=1.1,
            markersize=7,
            label="4SCM",
        ),
    ]

    fig.legend(
        handles=method_handles,
        loc="lower center",
        #bbox_to_anchor=(0.34, -0.01),
        bbox_to_anchor=(0.34, 0.045),
        ncol=3,
        frameon=False,
        fontsize=8,
        title="Method",
        title_fontsize=8,
    )

    # Beta legend: edge width
    beta_handles = []
    beta_show = sorted(set(beta_values))

    if len(beta_show) > 4:
        beta_show = [min(beta_show), np.median(beta_show), max(beta_show)]

    for b in beta_show:
        beta_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=_beta_to_linewidth(b, beta_min, beta_max),
                markersize=7,
                label=f"$\\beta={b:g}$",
            )
        )

    if beta_handles:
        fig.legend(
            handles=beta_handles,
            loc="lower center",
            #bbox_to_anchor=(0.67, -0.01),
            bbox_to_anchor=(0.67, 0.045),
            ncol=len(beta_handles),
            frameon=False,
            fontsize=8,
            title="$\\beta$",
            title_fontsize=8,
        )

    # Gamma colorbar
    if gamma_values:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        #cax = fig.add_axes([0.91, 0.23, 0.015, 0.55])
        cax = fig.add_axes([0.88, 0.24, 0.015, 0.65])#[0.88, 0.24, 0.015, 0.68]
        cbar = fig.colorbar(
            sm,
            ax=axes,
            orientation="vertical",
            cax=cax,
            fraction=0.025,
            #pad=0.04,#pad=0.015,
        )
        cbar.set_label("$\\gamma$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    #plt.tight_layout(rect=[0, 0.08, 0.93, 1]) # legenden position an 2. stelle 3. runter gehen für mehr platz an der gamma skala
    #plt.tight_layout(rect=[0, 0.10, 0.90, 1])
    plt.tight_layout(rect=[0, 0.12, 0.90, 1])
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_std" if show_std else ""
        fig.savefig(save_dir / f"pareto_matrix{suffix}.pdf", bbox_inches="tight")
        fig.savefig(save_dir / f"pareto_matrix{suffix}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
'''

def _positive_values(values):
    out = []
    for v in values:
        try:
            v = float(v)
            if np.isfinite(v) and v > 0:
                out.append(v)
        except Exception:
            pass
    return out


def _finite_values(values):
    out = []
    for v in values:
        try:
            v = float(v)
            if np.isfinite(v):
                out.append(v)
        except Exception:
            pass
    return out


def _make_gamma_norm(gamma_values, gamma_log=False):
    if gamma_log:
        vals = _positive_values(gamma_values)
        if len(vals) == 0:
            return mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        vmin, vmax = min(vals), max(vals)
        if vmin == vmax:
            vmax = vmin * 10.0
        return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    vals = _finite_values(gamma_values)
    if len(vals) == 0:
        return mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        vmax = vmin + 1.0

    return mpl.colors.Normalize(vmin=vmin, vmax=vmax)


def _scale_param(value, values, use_log=False):
    try:
        value = float(value)
    except Exception:
        return 0.0

    if not np.isfinite(value):
        return 0.0

    if use_log:
        vals = _positive_values(values)

        if value <= 0 or len(vals) == 0:
            return 0.0

        vmin, vmax = min(vals), max(vals)

        if vmin == vmax:
            return 1.0

        value = np.log10(value)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    else:
        vals = _finite_values(values)

        if len(vals) == 0:
            return 0.0

        vmin, vmax = min(vals), max(vals)

        if vmin == vmax:
            return 1.0

    return np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)


def _beta_to_linewidth_scaled(
    beta,
    beta_values,
    beta_log=False,
    lw_zero=0.6,
    lw_min=1.0,
    lw_max=3.2,
):
    """
    beta=0 is treated as a special case when beta_log=True:
    it gets lw_zero, while positive beta values are log-scaled.
    """
    try:
        beta = float(beta)
    except Exception:
        return lw_zero

    if not np.isfinite(beta):
        return lw_zero

    if beta_log and beta <= 0:
        return lw_zero

    t = _scale_param(beta, beta_values, use_log=beta_log)
    return lw_min + t * (lw_max - lw_min)



def _format_param_tick(v, latex=False):
    """Compact parameter label."""
    try:
        v = float(v)
    except Exception:
        return str(v)

    if not np.isfinite(v):
        return "--"
    if np.isclose(v, 0.0):
        return "$0$" if latex else "0"

    exp = int(np.round(np.log10(abs(v))))
    if np.isclose(v, 10.0 ** exp, rtol=1e-8, atol=0.0):
        return f"$10^{{{exp}}}$" if latex else f"10^{exp}"
    if np.isclose(v, 0.5):
        return "$5\\cdot10^{-1}$" if latex else "5e-1"
    return f"${v:g}$" if latex else f"{v:g}"

def plot_pareto_matrix(
        method_results,
        save_dir=None,
        show=True,
        show_std=False,
        gamma_cmap="Blues",
        gamma_log=True,
        beta_log=True,
):
    gamma_values = _finite_param_values(method_results, "gamma")
    beta_values = _finite_param_values(method_results, "beta")

    norm = _make_gamma_norm(gamma_values, gamma_log=gamma_log)
    #cmap = plt.get_cmap(gamma_cmap)
    base_cmap = plt.get_cmap(gamma_cmap)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{gamma_cmap}_cropped",
        base_cmap(np.linspace(0.2, 1.0, 256))
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

    for ax, panel, letter in zip(axes, PARETO_PAIRS, ["A", "B", "C"]):
        for mr in method_results:
            x = get_metric_value(mr, panel["x"])
            y = get_metric_value(mr, panel["y"])

            if x is None or y is None:
                continue

            family = mr["family"]
            marker = METHOD_MARKERS.get(family, "o")

            if family == "4SCM":
                facecolor = "0.72"
                edgecolor = "0.25"
                linewidth = 1.1

            else:
                gamma = mr.get("gamma", None)
                beta = mr.get("beta", None)

                try:
                    gamma = float(gamma)
                except Exception:
                    gamma = 0.0

                try:
                    beta = float(beta)
                except Exception:
                    beta = 0.0

                if gamma_log and gamma <= 0:
                    # gamma=0 is not log-scalable and represents no gamma regularization.
                    # Therefore it is visualized as a distinct white marker.
                    facecolor = "white"
                else:
                    facecolor = cmap(norm(gamma))

                edgecolor = "black"
                linewidth = _beta_to_linewidth_scaled(
                    beta,
                    beta_values,
                    beta_log=beta_log,
                    lw_zero=0.6,
                    lw_min=1.0,
                    lw_max=3.2,
                )

            if show_std:
                sx = get_metric_std(mr, panel["x"])
                sy = get_metric_std(mr, panel["y"])

                ax.errorbar(
                    x,
                    y,
                    xerr=sx,
                    yerr=sy,
                    fmt="none",
                    ecolor="0.45",
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.55,
                    zorder=1,
                )

            ax.scatter(
                x,
                y,
                s=80,
                marker=marker,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.95,
                zorder=3,
            )

        _add_preferred_direction(ax, panel["good_corner"])

        ax.set_title(f"({letter}) {panel['title']}", fontsize=10)
        ax.set_xlabel(panel["xlabel"], fontsize=9)
        ax.set_ylabel(panel["ylabel"], fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(axis="both", labelsize=8)
        if panel.get("invert_x", False):
            ax.invert_xaxis()

        if panel.get("invert_y", False):
            ax.invert_yaxis()

    # -------------------------------------------------------------------------
    # Method legend
    # -------------------------------------------------------------------------
    method_handles = [
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS["SPN"],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=7,
            label="SPN",
        ),
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS["MLP"],
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=7,
            label="MLP",
        ),
        plt.Line2D(
            [0], [0],
            marker=METHOD_MARKERS["4SCM"],
            linestyle="",
            markerfacecolor="0.72",
            markeredgecolor="0.25",
            markeredgewidth=1.1,
            markersize=7,
            label="4SCM",
        ),
    ]

    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.34, 0.045),
        ncol=3,
        frameon=False,
        fontsize=8,
        title="Method",
        title_fontsize=8,
    )

    # -------------------------------------------------------------------------
    # Beta legend: edge width
    # -------------------------------------------------------------------------
    beta_handles = []
    beta_show = sorted(set(_finite_values(beta_values)))

    if len(beta_show) > 4:
        positive_beta = _positive_values(beta_show)

        if beta_log and len(positive_beta) > 0:
            beta_show = []
            if 0.0 in set(beta_values):
                beta_show.append(0.0)
            beta_show.extend([min(positive_beta), np.median(positive_beta), max(positive_beta)])
            beta_show = sorted(set(beta_show))
        else:
            beta_show = [min(beta_show), np.median(beta_show), max(beta_show)]

    for b in beta_show:
        beta_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=_beta_to_linewidth_scaled(
                    b,
                    beta_values,
                    beta_log=beta_log,
                    lw_zero=0.6,
                    lw_min=1.0,
                    lw_max=3.2,
                ),
                markersize=7,
                label=f"$\\beta={_format_param_tick(b)}$",
            )
        )

    if beta_handles:
        fig.legend(
            handles=beta_handles,
            loc="lower center",
            bbox_to_anchor=(0.67, 0.045),
            ncol=len(beta_handles),
            frameon=False,
            fontsize=8,
            title="$\\beta$",
            title_fontsize=8,
        )

    # -------------------------------------------------------------------------
    # Gamma colorbar
    # -------------------------------------------------------------------------
    if gamma_values:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cax = fig.add_axes([0.89, 0.25, 0.015, 0.65])

        cbar = fig.colorbar(
            sm,
            cax=cax,
            orientation="vertical",
        )

        cbar.set_label("$\\gamma$", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        gamma_unique = sorted(set(_finite_values(gamma_values)))
        gamma_positive = sorted(set(_positive_values(gamma_values)))

        if gamma_log:
            cbar_ticks = gamma_positive
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([_format_param_tick(t) for t in cbar_ticks])

            if 0.0 in gamma_unique:
                # The zero case is shown as white markers and is not part of
                # the logarithmic colorbar.
                zero_handle = plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=1.0,
                    markersize=7,
                    label="$\\gamma=0$",
                )

                fig.legend(
                    handles=[zero_handle],
                    loc="lower center",
                    bbox_to_anchor=(0.87, 0.045),
                    frameon=False,
                    fontsize=8,
                )

        else:
            if len(gamma_unique) <= 6:
                cbar.set_ticks(gamma_unique)
                cbar.set_ticklabels([_format_param_tick(t) for t in gamma_unique])

    plt.tight_layout(rect=[0, 0.12, 0.90, 1])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_std" if show_std else ""
        suffix += "_glog" if gamma_log else ""
        suffix += "_blog" if beta_log else ""

        fig.savefig(save_dir / f"{img_id}pareto_matrix{suffix}.pdf", bbox_inches="tight")
        fig.savefig(save_dir / f"{img_id}pareto_matrix{suffix}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)






def collect_method_results_paper(results):
    out = []

    for item in results:
        if len(item) == 2:
            path, method_name = item
            beta = None
            gamma = None
            kld_weight = None

        elif len(item) == 5:
            path, method_name, method_suffix, beta, gamma = item
            kld_weight = str(method_suffix)
            method_name = f"{method_suffix}_{method_name}"

        else:
            raise ValueError(
                "Each entry must be either (path, method_name) "
                "or (path, method_name, method_suffix, beta, gamma)."
            )

        #metrics = load_metrics_json(str(path))
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print('load', path)
                metrics= json.load(f)
            family = infer_method_family(str(method_name))

            out.append({
                "method": str(method_name),
                "family": family,
                "path": str(path),
                "kld": kld_weight,
                "beta": beta,
                "gamma": gamma,
                "metrics": metrics,
                "aggregate_mean": metrics.get("aggregate_mean", {}),
                "aggregate_std": metrics.get("aggregate_std", {}),
            })
        else:
            print("NOT exist path")

    return out

def get_metric_std(method_result, metric_name):
    v = method_result.get("aggregate_std", {}).get(metric_name, None)
    if v is None:
        return None
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except Exception:
        return None

def _beta_to_linewidth(beta, beta_min, beta_max, lw_min=0.7, lw_max=3.2):
    if beta is None or not np.isfinite(beta):
        return 0.9

    if beta_max <= beta_min:
        return (lw_min + lw_max) / 2.0

    beta_norm = (float(beta) - beta_min) / (beta_max - beta_min)
    return lw_min + beta_norm * (lw_max - lw_min)


def _finite_param_values(method_results, key):
    vals = []
    for mr in method_results:
        if mr["family"] == "4SCM":
            continue
        v = mr.get(key, None)
        if v is None:
            continue
        try:
            v = float(v)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return vals

def collect_method_results_tab(
    results: Sequence[Union[Tuple[str, str], Tuple[str, str, str,float,float]]]
) -> List[Dict[str, Any]]:
    out = []
    for item in results:
        if len(item) == 2:
            path, method_name = item
            hue = ''
            beta,gamma=0,0
        elif len(item) == 5:
            path, method_name, hue,beta,gamma = item
        else:
            raise ValueError("Each result entry must be (path, method_name) or (path, method_name, hue_group).")

        #metrics = load_metrics_json(str(path))

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                print('load', path)
                metrics= json.load(f)
            print(path, flush=True)
            out.append({
                "method": hue+str(method_name),
                "hue": str(hue),
                "path": str(path),
                "metrics": metrics,
                'beta':beta,
                'gamma':gamma,
                "aggregate_mean": metrics.get("aggregate_mean", {}),
                "aggregate_std": metrics.get("aggregate_std", {}),
            })
    return out


def get_metric_value(method_result: Dict[str, Any], metric_name: str) -> Optional[float]:
    v = method_result.get("aggregate_mean", {}).get(metric_name, None)
    if v is None:
        return None
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def is_best_value(value: Optional[float], values: List[Optional[float]], metric_name: str, atol: float = 1e-12) -> bool:
    if value is None or not np.isfinite(value):
        return False

    valid = np.array([float(v) for v in values if v is not None and np.isfinite(v)], dtype=float)
    if valid.size == 0:
        return False

    better = METRIC_META.get(metric_name, {}).get("better", "up")
    if better == "down":
        best = np.min(valid)
        return abs(value - best) <= atol
    if better == "abs0":
        best = np.min(np.abs(valid))
        return abs(abs(value) - best) <= atol

    best = np.max(valid)
    return abs(value - best) <= atol


def maybe_bold(value_str: str, is_best: bool) -> str:
    return f"\\textbf{{{value_str}}}" if is_best and value_str != "--" else value_str


# =============================================================================
# LATEX TABLES
# =============================================================================

def build_latex_table(
    method_results: List[Dict[str, Any]],
    metric_names: List[str],
    caption: str,
    label: str,
    decimals: int = 5,
    use_resizebox: bool = True,
    bold_best: bool = True,
    show_std: bool = False,
) -> str:
    """
    Builds a LaTeX table where rows are metrics and columns are methods.

    The beta and gamma hyperparameters are always included as the first two
    rows. Metric cells contain means by default; with show_std=True, cells are
    formatted as mean ± std whenever aggregate_std is available. The best mean
    value per metric is bolded according to the metric direction.
    """
    methods = [latex_escape(m["method"]) for m in method_results]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    if use_resizebox:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    col_spec = "l" + "c" * len(methods)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    lines.append("Metric & " + " & ".join(methods) + "\\\\")
    lines.append("\\hline")

    beta_row = ["$\\beta$"] + [fmt_value(mr.get("beta", None), decimals=decimals) for mr in method_results]
    gamma_row = ["$\\gamma$"] + [fmt_value(mr.get("gamma", None), decimals=decimals) for mr in method_results]
    lines.append(" & ".join(beta_row) + "\\\\")
    lines.append(" & ".join(gamma_row) + "\\\\")
    lines.append("\\hline")

    for metric in metric_names:
        # Metric labels intentionally contain LaTeX arrows/math, so do not escape them here.
        row_name = display_name(metric)
        raw_values = [get_metric_value(mr, metric) for mr in method_results]
        row = [row_name]

        for mr, value in zip(method_results, raw_values):
            std = get_metric_std(mr, metric)
            value_str = fmt_mean_std(value, std, decimals=decimals, show_std=show_std)
            best = is_best_value(value, raw_values, metric) if bold_best else False
            row.append(maybe_bold(value_str, best))

        lines.append(" & ".join(row) + "\\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    if use_resizebox:
        lines.append("}%")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def build_main_text_explanation() -> str:
    return (
        "We report a compact set of complementary metrics that jointly assess target achievement, "
        "minimality, boundary-focused anatomical plausibility, and temporal consistency. CF target MAE "
        "and Spearman correlation quantify whether the generated counterfactuals achieve and follow the "
        "requested target values. Pixel $L_1$ minimality captures the magnitude of the image intervention. "
        "Boundary concentration, boundary contrast, and mean distance to the boundary evaluate whether "
        "changes occur near the LV contour, where physiologically plausible deformations are expected. "
        "Difficulty-weighted Dice summarizes anatomical preservation while accounting for the size of the "
        "requested target shift, and order preservation verifies that the ED/ES ordering of the two-frame "
        "sequence remains valid."
    )


# =============================================================================
# SCATTER PLOTS
# =============================================================================

def _get_metric_series(method_results: List[Dict[str, Any]], metric_name: str) -> List[float]:
    vals = []
    for mr in method_results:
        v = get_metric_value(mr, metric_name)
        vals.append(np.nan if v is None else float(v))
    return vals


def _make_hue_palette(hues: List[str], color_map: Optional[Dict[str, Any]] = None):
    unique = list(dict.fromkeys(hues))
    if color_map is not None:
        return {h: color_map.get(h, None) for h in unique}
    cmap = plt.get_cmap("tab10")
    return {h: cmap(i % 10) for i, h in enumerate(unique)}


def plot_main_scatter_panels(
    method_results: List[Dict[str, Any]],
    save_dir: Optional[str] = None,
    show: bool = True,
    hue_key: str = "hue",
    color_map: Optional[Dict[str, Any]] = None,
    annotate: bool = True,
):
    """
    Creates a 2x2 scatter panel. Colors are controlled by the hue value in
    method_results. If results were passed as (path, method), hue == method.
    If passed as (path, method, hue), identical hue values share the same color.
    """
    methods = [mr["method"] for mr in method_results]
    hues = [mr.get(hue_key, mr["method"]) for mr in method_results]
    palette = _make_hue_palette(hues, color_map=color_map)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.flatten()

    for ax, panel in zip(axes, PLOT_PAIRS):
        xvals = _get_metric_series(method_results, panel["x"])
        yvals = _get_metric_series(method_results, panel["y"])

        for x, y, method, hue in zip(xvals, yvals, methods, hues):
            if np.isfinite(x) and np.isfinite(y):
                ax.scatter(x, y, color=palette[hue], label=hue, s=60, alpha=0.9)
                if annotate:
                    ax.annotate(method, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.grid(True, alpha=0.3)

    # Single shared legend
    handles = []
    labels = []
    for hue in dict.fromkeys(hues):
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', color=palette[hue], label=hue))
        labels.append(hue)
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "main_scatter_panels.png", dpi=250, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_single_scatter(
    method_results: List[Dict[str, Any]],
    x_metric: str,
    y_metric: str,
    save_path: Optional[str] = None,
    show: bool = True,
    hue_key: str = "hue",
    color_map: Optional[Dict[str, Any]] = None,
    annotate: bool = True,
):
    methods = [mr["method"] for mr in method_results]
    hues = [mr.get(hue_key, mr["method"]) for mr in method_results]
    palette = _make_hue_palette(hues, color_map=color_map)
    xvals = _get_metric_series(method_results, x_metric)
    yvals = _get_metric_series(method_results, y_metric)

    fig, ax = plt.subplots(figsize=(6, 5))
    for x, y, method, hue in zip(xvals, yvals, methods, hues):
        if np.isfinite(x) and np.isfinite(y):
            ax.scatter(x, y, color=palette[hue], s=60, alpha=0.9)
            if annotate:
                ax.annotate(method, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlabel(display_name(x_metric))
    ax.set_ylabel(display_name(y_metric))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=250, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)




# =============================================================================
# AUTOMATED TRADE-OFF ANALYSIS
# =============================================================================

# Each trade-off is defined by metrics that are converted to a loss direction
# first. After min-max normalization, lower score is better.
TRADEOFF_DEFINITIONS = [
    {
        "name": "validity_minimality",
        "metrics": ["cf_validity_mae_ef", "pixel_l1_minimality"],
        "label": "Validity vs. minimality",
    },
    {
        "name": "validity_boundary_focus",
        "metrics": ["cf_validity_mae_ef", "pair_mean_boundary_concentration_strict"],
        "label": "Validity vs. LV boundary focus",
    },
    {
        "name": "minimality_boundary_focus",
        "metrics": ["pixel_l1_minimality", "pair_mean_boundary_concentration_strict"],
        "label": "Minimality vs. LV boundary focus",
    },

    #{
    #    "name": "validity_anatomy_preservation",
    #    "metrics": ["cf_validity_mae_ef", "pair_mean_dice_org_cf"],
    #    "label": "Validity vs. anatomy preservation",
    #},
]


def _as_float_or_nan(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def metric_to_loss_value(method_result: Dict[str, Any], metric_name: str) -> float:
    """Return metric value with orientation changed so that lower is better."""
    value = get_metric_value(method_result, metric_name)
    if value is None:
        return np.nan

    better = METRIC_META.get(metric_name, {}).get("better", "down")
    value = float(value)

    if better == "up":
        return -value
    if better == "abs0":
        return abs(value)
    return value


def _minmax_normalize(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return out
    vmin = np.nanmin(arr[mask])
    vmax = np.nanmax(arr[mask])
    if np.isclose(vmax, vmin):
        out[mask] = 0.0
    else:
        out[mask] = (arr[mask] - vmin) / (vmax - vmin)
    return out


def _group_key(method_result: Dict[str, Any], by_kld: bool = True) -> str:
    if not by_kld:
        return "all"
    kld = method_result.get("kld", None)
    return "all" if kld is None else str(kld)


def _rank_values(values: Sequence[float]) -> np.ndarray:
    """Simple ascending ranks; lower values receive lower ranks."""
    arr = np.array(values, dtype=float)
    ranks = np.full(arr.shape, np.nan, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return ranks
    order = np.argsort(arr[mask])
    tmp = np.empty(mask.sum(), dtype=float)
    tmp[order] = np.arange(1, mask.sum() + 1)
    ranks[mask] = tmp
    return ranks


def _spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    rx = _rank_values(x[mask])
    ry = _rank_values(y[mask])
    if np.nanstd(rx) == 0 or np.nanstd(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def is_pareto_optimal(loss_matrix: np.ndarray) -> np.ndarray:
    """Return boolean mask for Pareto-optimal rows. All objectives are minimized."""
    losses = np.array(loss_matrix, dtype=float)
    n = losses.shape[0]
    optimal = np.ones(n, dtype=bool)

    for i in range(n):
        if not np.all(np.isfinite(losses[i])):
            optimal[i] = False
            continue
        for j in range(n):
            if i == j or not np.all(np.isfinite(losses[j])):
                continue
            dominates = np.all(losses[j] <= losses[i]) and np.any(losses[j] < losses[i])
            if dominates:
                optimal[i] = False
                break
    return optimal


def compute_tradeoff_scores(
    method_results: List[Dict[str, Any]],
    tradeoffs: Sequence[Dict[str, Any]] = TRADEOFF_DEFINITIONS,
    normalize_within_kld: bool = True,
    include_families: Sequence[str] = ("MLP", "SPN"),
) -> List[Dict[str, Any]]:
    """Compute normalized scores and Pareto flags for each KLD/method/config.

    Normalization is min-max based and performed per KLD group by default. This
    avoids a KLD setting with a wider metric range dominating the aggregated
    score. Set normalize_within_kld=False for one global normalization.
    """
    rows = [mr for mr in method_results if mr.get("family") in include_families]
    groups = sorted(set(_group_key(mr, by_kld=normalize_within_kld) for mr in rows))
    out = []

    for group in groups:
        group_rows = [mr for mr in rows if _group_key(mr, by_kld=normalize_within_kld) == group]

        # Precompute normalized loss values per metric within the current group.
        all_metrics = sorted({m for tr in tradeoffs for m in tr["metrics"]})
        norm_losses = {}
        raw_losses = {}
        for metric in all_metrics:
            raw = [metric_to_loss_value(mr, metric) for mr in group_rows]
            raw_losses[metric] = raw
            norm_losses[metric] = _minmax_normalize(raw)

        for tradeoff in tradeoffs:
            metric_names = tradeoff["metrics"]
            loss_matrix = np.column_stack([raw_losses[m] for m in metric_names])
            pareto_flags = is_pareto_optimal(loss_matrix)

            for idx, mr in enumerate(group_rows):
                score_terms = [norm_losses[m][idx] for m in metric_names]
                valid_terms = [v for v in score_terms if np.isfinite(v)]
                score = float(np.mean(valid_terms)) if len(valid_terms) == len(metric_names) else np.nan

                row = {
                    "kld": mr.get("kld", "all"),
                    "method": mr.get("method"),
                    "family": mr.get("family"),
                    "beta": mr.get("beta"),
                    "gamma": mr.get("gamma"),
                    "tradeoff": tradeoff["name"],
                    "tradeoff_label": tradeoff.get("label", tradeoff["name"]),
                    "score": score,
                    "pareto_optimal": bool(pareto_flags[idx]),
                    "path": mr.get("path"),
                }
                for metric in metric_names:
                    row[f"raw_loss__{metric}"] = raw_losses[metric][idx]
                    row[f"norm_loss__{metric}"] = norm_losses[metric][idx]
                    row[f"metric__{metric}"] = get_metric_value(mr, metric)
                out.append(row)

    # Add ranks inside kld x tradeoff.
    for kld in sorted(set(str(r["kld"]) for r in out)):
        for tradeoff in sorted(set(r["tradeoff"] for r in out)):
            idxs = [i for i, r in enumerate(out) if str(r["kld"]) == kld and r["tradeoff"] == tradeoff]
            ranks = _rank_values([out[i]["score"] for i in idxs])
            for i, rank in zip(idxs, ranks):
                out[i]["rank"] = int(rank) if np.isfinite(rank) else None

    return out


def summarize_best_tradeoffs(
    score_rows: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Return top-k beta/gamma configurations per KLD, tradeoff, and method family."""
    out = []
    keys = sorted(set((str(r["kld"]), r["tradeoff"], r["family"]) for r in score_rows))

    for kld, tradeoff, family in keys:
        rows = [r for r in score_rows if str(r["kld"]) == kld and r["tradeoff"] == tradeoff and r["family"] == family]
        rows = [r for r in rows if np.isfinite(_as_float_or_nan(r.get("score")))]
        rows = sorted(rows, key=lambda r: r["score"])
        if not rows:
            continue
        best_score = rows[0]["score"]
        for local_rank, r in enumerate(rows[:top_k], start=1):
            out.append({
                "kld": kld,
                "tradeoff": tradeoff,
                "family": family,
                "local_rank": local_rank,
                "beta": r.get("beta"),
                "gamma": r.get("gamma"),
                "score": r.get("score"),
                "delta_to_best": r.get("score") - best_score,
                "pareto_optimal": r.get("pareto_optimal"),
                "global_rank_within_kld_tradeoff": r.get("rank"),
                "method": r.get("method"),
            })
    return out


def summarize_method_comparison(score_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare MLP and SPN per KLD/tradeoff using best, median, and Pareto counts."""
    out = []
    keys = sorted(set((str(r["kld"]), r["tradeoff"]) for r in score_rows))

    for kld, tradeoff in keys:
        rows = [r for r in score_rows if str(r["kld"]) == kld and r["tradeoff"] == tradeoff]
        summary = {"kld": kld, "tradeoff": tradeoff}
        for family in ["MLP", "SPN"]:
            fam_rows = [r for r in rows if r["family"] == family and np.isfinite(_as_float_or_nan(r.get("score")))]
            scores = np.array([r["score"] for r in fam_rows], dtype=float)
            summary[f"{family}_n"] = len(fam_rows)
            summary[f"{family}_best_score"] = float(np.min(scores)) if scores.size else np.nan
            summary[f"{family}_median_score"] = float(np.median(scores)) if scores.size else np.nan
            summary[f"{family}_pareto_count"] = int(sum(bool(r.get("pareto_optimal")) for r in fam_rows))
            summary[f"{family}_best_beta"] = None if not fam_rows else min(fam_rows, key=lambda r: r["score"]).get("beta")
            summary[f"{family}_best_gamma"] = None if not fam_rows else min(fam_rows, key=lambda r: r["score"]).get("gamma")

        summary["winner_best_score"] = (
            "SPN" if summary.get("SPN_best_score", np.inf) < summary.get("MLP_best_score", np.inf)
            else "MLP" if summary.get("MLP_best_score", np.inf) < summary.get("SPN_best_score", np.inf)
            else "tie"
        )
        summary["winner_median_score"] = (
            "SPN" if summary.get("SPN_median_score", np.inf) < summary.get("MLP_median_score", np.inf)
            else "MLP" if summary.get("MLP_median_score", np.inf) < summary.get("SPN_median_score", np.inf)
            else "tie"
        )
        summary["winner_pareto_count"] = (
            "SPN" if summary.get("SPN_pareto_count", 0) > summary.get("MLP_pareto_count", 0)
            else "MLP" if summary.get("MLP_pareto_count", 0) > summary.get("SPN_pareto_count", 0)
            else "tie"
        )
        out.append(summary)
    return out


def summarize_beta_gamma_sensitivity(
    method_results: List[Dict[str, Any]],
    metric_names: Sequence[str],
    include_families: Sequence[str] = ("MLP", "SPN"),
) -> List[Dict[str, Any]]:
    """Estimate whether beta/gamma are associated with each metric.

    The reported correlations are Spearman correlations between log10(1+x) of
    the parameter and the loss-oriented metric value. Positive values mean that
    increasing the parameter worsens the metric; negative values mean that it
    improves the metric.
    """
    rows = [mr for mr in method_results if mr.get("family") in include_families]
    out = []

    for kld in sorted(set(str(mr.get("kld", "all")) for mr in rows)):
        for family in include_families:
            fam_rows = [mr for mr in rows if str(mr.get("kld", "all")) == kld and mr.get("family") == family]
            if not fam_rows:
                continue
            beta = [np.log10(1.0 + max(0.0, _as_float_or_nan(mr.get("beta")))) for mr in fam_rows]
            gamma = [np.log10(1.0 + max(0.0, _as_float_or_nan(mr.get("gamma")))) for mr in fam_rows]

            for metric in metric_names:
                y = [metric_to_loss_value(mr, metric) for mr in fam_rows]
                beta_corr = _spearman_corr(beta, y)
                gamma_corr = _spearman_corr(gamma, y)

                stronger = "beta" if abs(beta_corr) > abs(gamma_corr) else "gamma"
                if not np.isfinite(beta_corr) and not np.isfinite(gamma_corr):
                    stronger = "undetermined"

                out.append({
                    "kld": kld,
                    "family": family,
                    "metric": metric,
                    "metric_label": display_name(metric),
                    "beta_spearman_loss": beta_corr,
                    "gamma_spearman_loss": gamma_corr,
                    "stronger_parameter": stronger,
                    "interpretation": (
                        "negative = larger parameter tends to improve metric; "
                        "positive = larger parameter tends to worsen metric"
                    ),
                })
    return out



# -----------------------------------------------------------------------------
# Compact and non-linear summaries for interpretability
# -----------------------------------------------------------------------------

def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean_finite(values: Sequence[float]) -> float:
    arr = _finite_array(values)
    return float(np.mean(arr)) if arr.size else np.nan


def _median_finite(values: Sequence[float]) -> float:
    arr = _finite_array(values)
    return float(np.median(arr)) if arr.size else np.nan


def _std_finite(values: Sequence[float]) -> float:
    arr = _finite_array(values)
    return float(np.std(arr)) if arr.size else np.nan


def _effect_direction_label(effect: float, threshold: float = 0.10) -> str:
    """Human-readable direction for loss-oriented effects/correlations."""
    if not np.isfinite(effect):
        return "undetermined"
    if effect <= -threshold:
        return "improves"
    if effect >= threshold:
        return "worsens"
    return "weak_or_mixed"


def _effect_strength_label(effect: float) -> str:
    """Qualitative magnitude label for correlations or variance shares."""
    if not np.isfinite(effect):
        return "undetermined"
    a = abs(float(effect))
    if a < 0.20:
        return "weak"
    if a < 0.50:
        return "moderate"
    return "strong"


def summarize_sensitivity_over_klds(sensitivity_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate the detailed Spearman sensitivity table over KLD weights.

    This keeps the sign convention of the detailed table: negative means that a
    larger parameter tends to improve the respective metric after orienting all
    metrics as losses; positive means worsening.
    """
    out = []
    keys = sorted(set((r.get("family"), r.get("metric")) for r in sensitivity_rows))
    for family, metric in keys:
        rows = [r for r in sensitivity_rows if r.get("family") == family and r.get("metric") == metric]
        beta_vals = [_as_float_or_nan(r.get("beta_spearman_loss")) for r in rows]
        gamma_vals = [_as_float_or_nan(r.get("gamma_spearman_loss")) for r in rows]
        beta_mean = _mean_finite(beta_vals)
        gamma_mean = _mean_finite(gamma_vals)
        beta_abs = _mean_finite([abs(v) for v in beta_vals])
        gamma_abs = _mean_finite([abs(v) for v in gamma_vals])
        if not np.isfinite(beta_abs) and not np.isfinite(gamma_abs):
            dominant = "undetermined"
        else:
            dominant = "beta" if beta_abs > gamma_abs else "gamma"
        out.append({
            "family": family,
            "metric": metric,
            "metric_label": display_name(metric),
            "n_kld": len(rows),
            "beta_spearman_mean": beta_mean,
            "gamma_spearman_mean": gamma_mean,
            "beta_abs_spearman_mean": beta_abs,
            "gamma_abs_spearman_mean": gamma_abs,
            "dominant_parameter_over_klds": dominant,
            "beta_direction": _effect_direction_label(beta_mean),
            "gamma_direction": _effect_direction_label(gamma_mean),
            "beta_strength": _effect_strength_label(beta_mean),
            "gamma_strength": _effect_strength_label(gamma_mean),
        })
    return out


def summarize_method_comparison_over_klds(method_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact MLP-vs-SPN summary aggregated over KLD weights per trade-off."""
    out = []
    for tradeoff in sorted(set(r.get("tradeoff") for r in method_rows)):
        rows = [r for r in method_rows if r.get("tradeoff") == tradeoff]
        n = len(rows)
        out.append({
            "tradeoff": tradeoff,
            "n_kld": n,
            "SPN_wins_best_score": sum(r.get("winner_best_score") == "SPN" for r in rows),
            "MLP_wins_best_score": sum(r.get("winner_best_score") == "MLP" for r in rows),
            "ties_best_score": sum(r.get("winner_best_score") == "tie" for r in rows),
            "SPN_wins_median_score": sum(r.get("winner_median_score") == "SPN" for r in rows),
            "MLP_wins_median_score": sum(r.get("winner_median_score") == "MLP" for r in rows),
            "ties_median_score": sum(r.get("winner_median_score") == "tie" for r in rows),
            "SPN_wins_pareto_count": sum(r.get("winner_pareto_count") == "SPN" for r in rows),
            "MLP_wins_pareto_count": sum(r.get("winner_pareto_count") == "MLP" for r in rows),
            "ties_pareto_count": sum(r.get("winner_pareto_count") == "tie" for r in rows),
            "mean_SPN_best_score": _mean_finite([_as_float_or_nan(r.get("SPN_best_score")) for r in rows]),
            "mean_MLP_best_score": _mean_finite([_as_float_or_nan(r.get("MLP_best_score")) for r in rows]),
            "mean_SPN_median_score": _mean_finite([_as_float_or_nan(r.get("SPN_median_score")) for r in rows]),
            "mean_MLP_median_score": _mean_finite([_as_float_or_nan(r.get("MLP_median_score")) for r in rows]),
            "mean_SPN_pareto_count": _mean_finite([_as_float_or_nan(r.get("SPN_pareto_count")) for r in rows]),
            "mean_MLP_pareto_count": _mean_finite([_as_float_or_nan(r.get("MLP_pareto_count")) for r in rows]),
        })
    return out


def summarize_best_config_frequency(best_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Count which beta/gamma pairs occur most often among local rank-1 configs."""
    rank1 = [r for r in best_rows if int(r.get("local_rank", -1)) == 1]
    out = []
    keys = sorted(set((r.get("tradeoff"), r.get("family"), r.get("beta"), r.get("gamma")) for r in rank1))
    for tradeoff, family, beta, gamma in keys:
        rows = [r for r in rank1 if r.get("tradeoff") == tradeoff and r.get("family") == family
                and r.get("beta") == beta and r.get("gamma") == gamma]
        out.append({
            "tradeoff": tradeoff,
            "family": family,
            "beta": beta,
            "gamma": gamma,
            "rank1_count_over_klds": len(rows),
            "klds_where_best": ";".join(sorted(str(r.get("kld")) for r in rows)),
            "mean_best_score_when_selected": _mean_finite([_as_float_or_nan(r.get("score")) for r in rows]),
        })
    out = sorted(out, key=lambda r: (r["tradeoff"], r["family"], -r["rank1_count_over_klds"], r["beta"], r["gamma"]))
    return out


def summarize_parameter_level_effects(
    method_results: List[Dict[str, Any]],
    metric_names: Sequence[str],
    include_families: Sequence[str] = ("MLP", "SPN"),
    parameter_names: Sequence[str] = ("beta", "gamma"),
) -> List[Dict[str, Any]]:
    """Non-linear sweet-spot summary treating beta/gamma as categorical levels.

    For every family/metric/parameter, this reports the best parameter level
    after averaging the loss-oriented metric over KLDs and over the other
    parameter. This detects U-shaped or threshold-like effects that Spearman can
    miss.
    """
    rows = [mr for mr in method_results if mr.get("family") in include_families]
    out = []

    for family in include_families:
        fam_rows = [mr for mr in rows if mr.get("family") == family]
        for metric in metric_names:
            for param in parameter_names:
                levels = sorted(set(_as_float_or_nan(mr.get(param)) for mr in fam_rows))
                levels = [v for v in levels if np.isfinite(v)]
                if not levels:
                    continue
                level_summaries = []
                for level in levels:
                    level_rows = [mr for mr in fam_rows if np.isclose(_as_float_or_nan(mr.get(param)), level)]
                    vals = [metric_to_loss_value(mr, metric) for mr in level_rows]
                    level_summaries.append((level, _mean_finite(vals), len(_finite_array(vals))))
                finite = [(lvl, mean, n) for lvl, mean, n in level_summaries if np.isfinite(mean)]
                if not finite:
                    continue
                best_level, best_mean, best_n = min(finite, key=lambda x: x[1])
                worst_level, worst_mean, worst_n = max(finite, key=lambda x: x[1])
                spread = worst_mean - best_mean
                # compact level curve, ordered by level
                curve = ";".join(f"{lvl:g}:{mean:.6g}" for lvl, mean, n in finite)
                out.append({
                    "family": family,
                    "metric": metric,
                    "metric_label": display_name(metric),
                    "parameter": param,
                    "best_level": best_level,
                    "worst_level": worst_level,
                    "best_mean_loss": best_mean,
                    "worst_mean_loss": worst_mean,
                    "loss_spread_worst_minus_best": spread,
                    "n_at_best_level": best_n,
                    "level_mean_loss_curve": curve,
                    "interpretation": "lowest mean loss is best; levels are categorical, so this can reveal non-linear sweet spots",
                })
    return out


def _variance_explained_by_group(values: Sequence[float], groups: Sequence[Any]) -> float:
    """Return Var(E[y|group]) / Var(y), using loss-oriented y values."""
    y = np.array(values, dtype=float)
    g = np.array(groups, dtype=object)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]
    g = g[mask]
    total_var = float(np.var(y))
    if np.isclose(total_var, 0.0):
        return 0.0
    group_means = []
    group_weights = []
    for level in sorted(set(g), key=lambda x: str(x)):
        vals = y[g == level]
        if vals.size:
            group_means.append(float(np.mean(vals)))
            group_weights.append(int(vals.size))
    group_means = np.array(group_means, dtype=float)
    group_weights = np.array(group_weights, dtype=float)
    weighted_mean = float(np.average(group_means, weights=group_weights))
    between = float(np.average((group_means - weighted_mean) ** 2, weights=group_weights))
    return between / total_var


def summarize_variance_decomposition(
    method_results: List[Dict[str, Any]],
    metric_names: Sequence[str],
    include_families: Sequence[str] = ("MLP", "SPN"),
) -> List[Dict[str, Any]]:
    """Non-linear effect-size summary for beta, gamma, and KLD.

    The reported share is Var(E[loss | factor]) / Var(loss). Unlike a linear
    correlation, it can capture non-monotone effects, e.g. an intermediate gamma
    being optimal.
    """
    rows = [mr for mr in method_results if mr.get("family") in include_families]
    out = []
    for family in include_families:
        fam_rows = [mr for mr in rows if mr.get("family") == family]
        for metric in metric_names:
            y = [metric_to_loss_value(mr, metric) for mr in fam_rows]
            beta_groups = [_as_float_or_nan(mr.get("beta")) for mr in fam_rows]
            gamma_groups = [_as_float_or_nan(mr.get("gamma")) for mr in fam_rows]
            kld_groups = [str(mr.get("kld", "all")) for mr in fam_rows]
            beta_share = _variance_explained_by_group(y, beta_groups)
            gamma_share = _variance_explained_by_group(y, gamma_groups)
            kld_share = _variance_explained_by_group(y, kld_groups)
            shares = {"beta": beta_share, "gamma": gamma_share, "kld": kld_share}
            dominant = max(shares, key=lambda k: -np.inf if not np.isfinite(shares[k]) else shares[k])
            out.append({
                "family": family,
                "metric": metric,
                "metric_label": display_name(metric),
                "beta_variance_share": beta_share,
                "gamma_variance_share": gamma_share,
                "kld_variance_share": kld_share,
                "dominant_factor": dominant,
                "interpretation": "higher share = stronger non-linear association with the metric; metrics are loss-oriented",
            })
    return out


def build_tradeoff_readme(prefix: str) -> str:
    return f"""Automated counterfactual trade-off analysis outputs for prefix: {prefix}

Core detailed outputs:
- {prefix}_all_scores.csv: all beta/gamma/KLD/method configurations with normalized trade-off scores and Pareto flags.
- {prefix}_best_configs.csv: top configurations per KLD, trade-off, and method.
- {prefix}_method_comparison.csv: MLP-vs-SPN comparison per KLD and trade-off.
- {prefix}_beta_gamma_sensitivity.csv: detailed Spearman sensitivity per KLD/family/metric.

Compact interpretability outputs:
- {prefix}_method_comparison_over_klds.csv: aggregates MLP-vs-SPN wins over all KLD settings.
- {prefix}_best_config_frequency.csv: shows which beta/gamma pairs most often become rank-1 across KLDs.
- {prefix}_sensitivity_over_klds.csv: compact Spearman summary aggregated over KLDs.
- {prefix}_parameter_level_effects.csv: categorical beta/gamma level summary; useful for non-linear sweet spots.
- {prefix}_variance_decomposition.csv: non-linear effect size Var(E[loss|factor]) / Var(loss) for beta, gamma, and KLD.

Sign convention:
All metrics are converted to a loss direction before scoring or sensitivity analysis. Therefore lower is always better. For Spearman correlations, negative means that increasing the parameter tends to improve the metric; positive means worsening. For variance decomposition, larger shares indicate stronger association but not whether the effect is good or bad.
"""

def write_dict_rows_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_tradeoff_analysis(
    results: Sequence[Union[Tuple[str, str], Tuple[str, str, str, float, float]]],
    save_dir: Optional[str] = None,
    prefix: str = "tradeoff",
    tradeoffs: Sequence[Dict[str, Any]] = TRADEOFF_DEFINITIONS,
    normalize_within_kld: bool = True,
    top_k: int = 3,
) -> Dict[str, Any]:
    """High-level driver for automated beta/gamma/KLD trade-off evaluation."""
    method_results = collect_method_results_paper(results)
    method_results = [mr for mr in method_results if mr.get("family") in ("MLP", "SPN")]

    score_rows = compute_tradeoff_scores(
        method_results,
        tradeoffs=tradeoffs,
        normalize_within_kld=normalize_within_kld,
        include_families=("MLP", "SPN"),
    )
    best_rows = summarize_best_tradeoffs(score_rows, top_k=top_k)
    method_rows = summarize_method_comparison(score_rows)

    sensitivity_metrics = sorted({m for tr in tradeoffs for m in tr["metrics"]})
    sensitivity_rows = summarize_beta_gamma_sensitivity(method_results, sensitivity_metrics)

    # Compact summaries that are easier to interpret than the detailed tables.
    method_over_kld_rows = summarize_method_comparison_over_klds(method_rows)
    best_frequency_rows = summarize_best_config_frequency(best_rows)
    sensitivity_over_kld_rows = summarize_sensitivity_over_klds(sensitivity_rows)
    parameter_level_rows = summarize_parameter_level_effects(method_results, sensitivity_metrics)
    variance_rows = summarize_variance_decomposition(method_results, sensitivity_metrics)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        write_dict_rows_csv(score_rows, save_dir / f"{prefix}_all_scores.csv")
        write_dict_rows_csv(best_rows, save_dir / f"{prefix}_best_configs.csv")
        write_dict_rows_csv(method_rows, save_dir / f"{prefix}_method_comparison.csv")
        write_dict_rows_csv(sensitivity_rows, save_dir / f"{prefix}_beta_gamma_sensitivity.csv")

        # New compact/interpretable outputs.
        write_dict_rows_csv(method_over_kld_rows, save_dir / f"{prefix}_method_comparison_over_klds.csv")
        write_dict_rows_csv(best_frequency_rows, save_dir / f"{prefix}_best_config_frequency.csv")
        write_dict_rows_csv(sensitivity_over_kld_rows, save_dir / f"{prefix}_sensitivity_over_klds.csv")
        write_dict_rows_csv(parameter_level_rows, save_dir / f"{prefix}_parameter_level_effects.csv")
        write_dict_rows_csv(variance_rows, save_dir / f"{prefix}_variance_decomposition.csv")
        (save_dir / f"{prefix}_README.txt").write_text(build_tradeoff_readme(prefix), encoding="utf-8")

    return {
        "method_results": method_results,
        "scores": score_rows,
        "best_configs": best_rows,
        "method_comparison": method_rows,
        "sensitivity": sensitivity_rows,
        "method_comparison_over_klds": method_over_kld_rows,
        "best_config_frequency": best_frequency_rows,
        "sensitivity_over_klds": sensitivity_over_kld_rows,
        "parameter_level_effects": parameter_level_rows,
        "variance_decomposition": variance_rows,
    }


# =============================================================================
# HIGH-LEVEL DRIVER
# =============================================================================

def generate_paper_outputs(
    results: Sequence[Union[Tuple[str, str],
    Tuple[str, str, str,float,float]]],
    save_dir: Optional[str] = None,
    decimals: int = 3,
    show_plots: bool = True,
    show_std=False,
    show_table_std: bool = False,
    gamma_cmap="Blues",
    color_map: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    #method_results = collect_method_results_tab(results)
    method_results = collect_method_results_paper(results)
    print('Collected',flush=True)
    main_table_tex = build_latex_table(
        method_results=method_results,
        metric_names=MAIN_METRICS,
        caption=(
            "Main counterfactual evaluation results. Arrows indicate whether higher or lower values are preferable; "
            "best values are highlighted in bold."
        ),
        label="tab:main_cf_eval",
        decimals=decimals,
        use_resizebox=True,
        bold_best=True,
        show_std=show_table_std,
    )
    print(main_table_tex)
    print('Appendix', flush=True)
    appendix_table_tex = build_latex_table(
        method_results=method_results,
        metric_names=APPENDIX_METRICS,
        caption=(
            "Extended counterfactual evaluation results including supporting localization and temporal metrics. "
            "Best values are highlighted in bold."
        ),
        label="tab:appendix_cf_eval",
        decimals=decimals,
        use_resizebox=True,
        bold_best=True,
        show_std=show_table_std,
    )
    print('Save', flush=True)
    explanation_text = build_main_text_explanation()



    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / f"{img_id}main_table.tex").write_text(main_table_tex, encoding="utf-8")
        (save_dir / f"{img_id}appendix_table.tex").write_text(appendix_table_tex, encoding="utf-8")
        (save_dir / f"{img_id}main_metrics_explanation.txt").write_text(explanation_text, encoding="utf-8")
    #method_results = collect_method_results(results)
    print('scatter', flush=True)

    #plot_main_scatter_panels(method_results, save_dir=save_dir, show=show_plots, color_map=color_map)
    #'''
    plot_pareto_matrix(
        method_results=method_results,
        save_dir=save_dir,
        show=show_plots,
        show_std=show_std,
        gamma_cmap=gamma_cmap,
    )
    #'''
    return {
        "main_table_tex": main_table_tex,
        "appendix_table_tex": appendix_table_tex,
        "explanation_text": explanation_text,
    }



# =============================================================================
# EXTENDED PARETO / PARAMETER SWEEP PLOTS
# =============================================================================

def _safe_float(x, default=np.nan):
    try:
        x = float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _unique_sorted_floats(values):
    vals = []
    for v in values:
        v = _safe_float(v)
        if np.isfinite(v):
            vals.append(v)
    return sorted(set(vals))


def _get_kld_values(method_results):
    vals = []
    for mr in method_results:
        if mr.get("family") == "4SCM":
            continue
        kld = mr.get("kld", None)
        if kld is not None:
            vals.append(str(kld))
    # Numerical-looking KLD labels are sorted numerically, otherwise lexically.
    def _sort_key(x):
        try:
            return (0, float(x))
        except Exception:
            return (1, str(x))
    return sorted(set(vals), key=_sort_key)


def _scatter_pareto_points(
    ax,
    rows,
    panel,
    beta_values,
    gamma_values,
    cmap,
    norm,
    show_std=False,
    gamma_log=True,
    beta_log=True,
    include_families=("MLP", "SPN"),
):
    """Shared low-level scatter routine for Pareto plots."""
    for mr in rows:
        family = mr.get("family")
        if family not in include_families:
            continue

        x = get_metric_value(mr, panel["x"])
        y = get_metric_value(mr, panel["y"])
        if x is None or y is None:
            continue

        marker = METHOD_MARKERS.get(family, "o")
        gamma = _safe_float(mr.get("gamma", 0.0), 0.0)
        beta = _safe_float(mr.get("beta", 0.0), 0.0)

        if gamma_log and gamma <= 0:
            facecolor = "white"
        else:
            facecolor = cmap(norm(gamma))

        linewidth = _beta_to_linewidth_scaled(
            beta,
            beta_values,
            beta_log=beta_log,
            lw_zero=0.7,
            lw_min=1.1,
            lw_max=3.6,
        )

        if show_std:
            sx = get_metric_std(mr, panel["x"])
            sy = get_metric_std(mr, panel["y"])
            ax.errorbar(
                x, y,
                xerr=sx,
                yerr=sy,
                fmt="none",
                ecolor="0.45",
                elinewidth=0.7,
                capsize=1.8,
                alpha=0.45,
                zorder=1,
            )

        ax.scatter(
            x,
            y,
            s=62,
            marker=marker,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=linewidth,
            alpha=0.95,
            zorder=3,
        )


def _add_method_legend_no_4scm(fig, bbox_to_anchor=(0.24, 0.035)):
    handles = [
        plt.Line2D(
            [0], [0], marker=METHOD_MARKERS["SPN"], linestyle="",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.2, markersize=7, label="SPN",
        ),
        plt.Line2D(
            [0], [0], marker=METHOD_MARKERS["MLP"], linestyle="",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.2, markersize=7, label="MLP",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=2,
        frameon=False,
        fontsize=8,
        title="Method",
        title_fontsize=8,
    )



def _add_beta_legend(
    fig,
    beta_values,
    beta_log=True,
    bbox_to_anchor=(0.50, 0.035),
    include_zero=False,
    max_cols=6,
):
    """Legend for beta as marker edge width; shows all positive beta levels."""
    beta_show = sorted(set(_finite_values(beta_values)))
    if not beta_show:
        return
    if not include_zero:
        beta_show = [b for b in beta_show if not np.isclose(b, 0.0)]

    handles = []
    for b in beta_show:
        lab = _format_param_tick(b, latex=True).replace("$", "")
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor="white", markeredgecolor="black",
                markeredgewidth=_beta_to_linewidth_scaled(
                    b, beta_values, beta_log=beta_log,
                    lw_zero=0.7, lw_min=0.9, lw_max=3.2,
                ),
                markersize=6.5,
                label=f"$\\beta={lab}$",
            )
        )

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=min(len(handles), max_cols),
        frameon=False,
        fontsize=7,
        title="$\\beta$",
        title_fontsize=7,
        handletextpad=0.45,
        columnspacing=0.75,
        borderaxespad=0.0,
    )


def _add_gamma_colorbar_and_zero(
    fig,
    cmap,
    norm,
    gamma_values,
    gamma_log=True,
    cax_rect=(0.905, 0.32, 0.009, 0.42),
    zero_bbox=None,
    show_zero_legend=False,
):
    """Add a compact gamma colorbar; gamma=0 is not shown as legend item."""
    if not gamma_values:
        return

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes(cax_rect)
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label("$\\gamma$", fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)

    gamma_unique = sorted(set(_finite_values(gamma_values)))
    gamma_positive = sorted(set(_positive_values(gamma_values)))

    if gamma_log:
        cbar.set_ticks(gamma_positive)
        #cbar.set_ticklabels([_format_param_tick(t, latex=True).replace("$", "") for t in gamma_positive])
        cbar.set_ticklabels([_format_param_tick(t, latex=True) for t in gamma_positive])
    else:
        ticks = gamma_positive if 0.0 in gamma_unique else gamma_unique
        if len(ticks) <= 7:
            cbar.set_ticks(ticks)
            #cbar.set_ticklabels([_format_param_tick(t, latex=True).replace("$", "") for t in ticks])
            cbar.set_ticklabels([_format_param_tick(t, latex=True) for t in ticks])


def plot_pareto_matrix_all_klds(
    method_results,
    save_dir=None,
    show=True,
    show_std=False,
    gamma_cmap="Blues",
    gamma_log=True,
    beta_log=True,
    kld_values=None,
    include_families=("MLP", "SPN"),
    add_preferred=True,
    filename_prefix="pareto_matrix_all_klds",
    figsize=(10.8, 7.0),
):
    """Compact 3x3 Pareto matrix: rows=KLD settings, columns=trade-offs."""
    if kld_values is None:
        kld_values = _get_kld_values(method_results)
    kld_values = [str(k) for k in kld_values]

    gamma_values = _finite_param_values(method_results, "gamma")
    beta_values = _finite_param_values(method_results, "beta")
    norm = _make_gamma_norm(gamma_values, gamma_log=gamma_log)

    base_cmap = plt.get_cmap(gamma_cmap)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{gamma_cmap}_cropped_allkld_v7",
        base_cmap(np.linspace(0.25, 0.95, 256)),
    )

    n_rows = len(kld_values)
    n_cols = len(PARETO_PAIRS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=False, sharey=False)

    for r, kld in enumerate(kld_values):
        rows = [mr for mr in method_results if str(mr.get("kld")) == str(kld)]
        for c, panel in enumerate(PARETO_PAIRS):
            ax = axes[r, c]
            _scatter_pareto_points(
                ax, rows, panel, beta_values, gamma_values, cmap, norm,
                show_std=show_std, gamma_log=gamma_log, beta_log=beta_log,
                include_families=include_families,
            )
            if r == 0:
                ax.set_title(panel["title"], fontsize=8, pad=4)
            if add_preferred:
                _add_preferred_direction(ax, panel["good_corner"])

            #if c == 0:
            #    ax.set_ylabel(f"$\\beta_1=10^{{-{kld}}}$\n{panel['ylabel']}", fontsize=8)
            #else:
            #    ax.set_ylabel("")

            #if c == 0:
            ax.set_ylabel(panel["ylabel"], fontsize=8)#, labelpad=10
            #else:
            #    ax.set_ylabel(panel["ylabel"], fontsize=8, labelpad=8)
            if c == 0:
                ax.text(
                    -0.24, 0.5,
                    rf"$\beta_1=10^{{-{kld}}}$",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                )



            if r == n_rows - 1:
                ax.set_xlabel(panel["xlabel"], fontsize=8)
            else:
                ax.set_xlabel("")

            ax.grid(True, alpha=0.22, linewidth=0.55)
            ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
            if panel.get("invert_x", False):
                ax.invert_xaxis()
            if panel.get("invert_y", False):
                ax.invert_yaxis()

    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.145, top=0.985, wspace=0.25, hspace=0.18) # w space =18

    _add_method_legend_no_4scm(fig, bbox_to_anchor=(0.22, 0.035))
    _add_beta_legend(fig, beta_values, beta_log=beta_log, bbox_to_anchor=(0.56, 0.035), include_zero=False, max_cols=5)
    _add_gamma_colorbar_and_zero(
        fig, cmap, norm, gamma_values, gamma_log=gamma_log,
        cax_rect=(0.91, 0.33, 0.008, 0.40), show_zero_legend=False,
    )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_std" if show_std else ""
        suffix += "_glog" if gamma_log else ""
        suffix += "_blog" if beta_log else ""
        fig.savefig(save_dir / f"{filename_prefix}{suffix}.pdf", bbox_inches="tight")
        fig.savefig(save_dir / f"{filename_prefix}{suffix}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def _select_parameter_sweep_rows(rows, sweep_name, level, atol=1e-12):
    """Return rows for one of three 1D sweeps through the beta/gamma grid.

    sweep_name == "beta": beta varies, gamma == 0.
    sweep_name == "gamma": gamma varies, beta == 0.
    sweep_name == "beta=gamma": beta == gamma varies.
    """
    selected = []
    for mr in rows:
        beta = _safe_float(mr.get("beta", np.nan))
        gamma = _safe_float(mr.get("gamma", np.nan))
        if not np.isfinite(beta) or not np.isfinite(gamma):
            continue

        if sweep_name == "beta":
            ok = np.isclose(gamma, 0.0, atol=atol) and np.isclose(beta, level, atol=atol)
        elif sweep_name == "gamma":
            ok = np.isclose(beta, 0.0, atol=atol) and np.isclose(gamma, level, atol=atol)
        elif sweep_name == "beta=gamma":
            ok = np.isclose(beta, gamma, atol=atol) and np.isclose(beta, level, atol=atol)
        else:
            ok = False

        if ok:
            selected.append(mr)
    return selected


def _metric_axis_label(metric_name):
    return display_name(metric_name)



def plot_parameter_tradeoff_sweeps_all_klds(
    method_results,
    save_dir=None,
    show=True,
    kld_values=None,
    include_families=("MLP", "SPN"),
    metrics=("cf_validity_mae_ef", "pixel_l1_minimality", "pair_mean_boundary_concentration_strict"),
    filename_prefix="parameter_tradeoff_sweeps_all_klds",
    figsize=(10.8, 7.0),
):
    """Parameter sweeps with one y-axis per subplot.

    Rows are metrics, columns are KLD settings. Each subplot contains MLP/SPN
    lines for beta-only, gamma-only, and beta=gamma sweeps.
    """
    if kld_values is None:
        kld_values = _get_kld_values(method_results)
    kld_values = [str(k) for k in kld_values]

    all_levels = sorted(set(
        _unique_sorted_floats([mr.get("beta") for mr in method_results]) +
        _unique_sorted_floats([mr.get("gamma") for mr in method_results])
    ))
    level_to_x = {v: i for i, v in enumerate(all_levels)}
    x_ticks = [level_to_x[v] for v in all_levels]
    #x_labels = [_format_param_tick(v, latex=True).replace("$", "") for v in all_levels]
    x_labels = [_format_param_tick(v, latex=True) for v in all_levels]

    family_colors = {"MLP": plt.get_cmap("tab10")(0), "SPN": plt.get_cmap("tab10")(1)}
    sweep_styles = {"beta": "-", "gamma": "--", "beta=gamma": ":"}
    sweep_markers = {"beta": "o", "gamma": "s", "beta=gamma": "D"}

    n_rows = len(metrics)
    n_cols = len(kld_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=True, sharey=False)

    for c, kld in enumerate(kld_values):
        kld_rows = [mr for mr in method_results if str(mr.get("kld")) == str(kld) and mr.get("family") in include_families]
        for r, metric in enumerate(metrics):
            ax = axes[r, c]
            for family in include_families:
                fam_rows = [mr for mr in kld_rows if mr.get("family") == family]
                color = family_colors.get(family, "0.25")
                for sweep_name, linestyle in sweep_styles.items():
                    xs, ys = [], []
                    for level in all_levels:
                        selected = _select_parameter_sweep_rows(fam_rows, sweep_name, level)
                        if not selected:
                            continue
                        val = _mean_finite([get_metric_value(mr, metric) for mr in selected])
                        if np.isfinite(val):
                            xs.append(level_to_x[level])
                            ys.append(val)
                    if xs:
                        ax.plot(xs, ys, color=color, linestyle=linestyle, marker=sweep_markers[sweep_name], markersize=3.0, linewidth=1.15, alpha=0.90)

            if r == 0:
                ax.set_title(f"$\\beta_1=10^{{-{kld}}}$", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(_metric_axis_label(metric), fontsize=8)
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("Regularization value", fontsize=8)
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels, rotation=0)
            else:
                ax.set_xlabel("")
                ax.set_xticks(x_ticks)
                ax.set_xticklabels([])

            ax.grid(True, alpha=0.22, linewidth=0.55)
            ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)

    family_handles = [plt.Line2D([0], [0], color=family_colors[f], lw=1.8, label=f) for f in include_families if f in family_colors]
    sweep_handles = [plt.Line2D([0], [0], color="0.25", lw=1.3, linestyle=ls, marker=sweep_markers[lab], markersize=3.2, label=lab) for lab, ls in sweep_styles.items()]

    fig.legend(handles=family_handles, loc="lower center", bbox_to_anchor=(0.30, 0.035), ncol=len(family_handles), frameon=False, fontsize=7, title="Method", title_fontsize=7, handletextpad=0.5, columnspacing=0.9)
    fig.legend(handles=sweep_handles, loc="lower center", bbox_to_anchor=(0.65, 0.035), ncol=3, frameon=False, fontsize=7, title="Regularization parameter", title_fontsize=7, handletextpad=0.5, columnspacing=0.9)

    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.145, top=0.96, wspace=0.22, hspace=0.08)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{filename_prefix}.pdf", bbox_inches="tight")
        fig.savefig(save_dir / f"{filename_prefix}.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    SHOW_STD_IN_PARETO = False
    SHOW_STD_IN_TABLES = True
    ds='stan_debug'

    project_path='/lustre/project/ki-topml/jsiekiera/cnn_spn_models/'
    spn_names=[#('stan_KLD4_MS_SSIM_HUBER_64_relu_DS_flip','k4_s1_f'),
               #('stan_KLD4_MS_SSIM_HUBER_64_relu_DS_no_flip','k4_s1_nf'),
               #('stan_KLD4_MS_SSIM_HUBER_64_relu_DS_2stage_flip','k4_s2_f'),
               #('stan_KLD5_MS_SSIM_HUBER_64_relu_DS_2stage_flip','k5_s2_f'),
               ###('stan_KLD4_MS_SSIM_HUBER_64_relu_DS_2stage_noflip','k4'), #'k4_s2_nf'
                #('KLD4_MS_SSIM_HUBER_4_stanford_DS_1stage_old',''),
                #('KLD4_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_old',''),
                #('KLD4_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_encrefin',''),

                ('KLD4_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_encrefinchec','4'),
                ('KLD5_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_encrefinchec', '5'),
                ('KLD6_MS_SSIM_HUBER_3_stanford_seg_DS_1stage_encrefinchec', '6'),

                #('KLD4_MS_SSIM_HUBER_3_CAMUS_DS_1stage_encrefinchec_tmp2', '4'),
                #('KLD5_MS_SSIM_HUBER_3_CAMUS_DS_1stage_encrefinchec_tmp2', '5'),
                #('KLD6_MS_SSIM_HUBER_3_CAMUS_DS_1stage_encrefinchec_tmp2', '6'),

                #('stan_KLD4_MS_SSIM_HUBER_64_relu_DS_2stage_enccheck_nn_mask_sh_1__52','k4')
                #('stan_KLD3_MS_SSIM_HUBER_64_relu_DS_2stage_enccheck_nn_mask_sh_1__52', 'k3'),
                #('stanseg_KLD4_MS_SSIM_HUBER_64_relu_DS_2stage_enccheck_nn_mask_sh_1__52', 'k4'),
                #('stanseg_KLD3_MS_SSIM_HUBER_64_relu_DS_2stage_enccheck_nn_mask_sh_1__52', 'k3')

               #('stan_KLD5_MS_SSIM_HUBER_64_relu_DS_2stage_noflip','k5')
               ] #'k5_s2_nf'
    #TODO consider the huename in the plot of generate_paper_outputs
    grid_str='/grid0/'
    results=[
        #('/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion/4SCM_merge/eval/counterfactual_videos/segment_analyse.json', "4SCM","v2"),# org stanford
        #('/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_merge/eval/counterfactual_videos/segment_analyse_.json',"4SCM"), #, "v3")
        #('/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_v3_CAMUS_merge/eval/counterfactual_videos/segment_analyse_CAMUS_flip.json',"4SCM")#segment_analyse_CAMUS_flip.json
        #('/fshpc/jsiekier/promotion/SPN_related_work/EchoDiffusion/outputs/diffusion3/4SCM_v3_CAMUS_merge/eval/counterfactual_videos/segment_analyse_.json',"4SCM")
    ]

    img_id=f'{ds}_largeSPN_all'
    for mod_path,mod_name in spn_names:
        results.extend([(project_path + mod_path + grid_str + "MLPb0-0g0-0-seg0mean.json", "MLP0", mod_name, 0, 0),
                        (project_path + mod_path + grid_str + "SPNb0-0g0-0-seg1mean.json", "SPN0", mod_name, 0, 0),
                        (project_path + mod_path + grid_str + "MLPb0-01g0-01-seg0mean.json", "MLP", mod_name, 0.01, 0.01),
                        (project_path + mod_path + grid_str + "SPNb0-01g0-01-seg1mean.json", "SPN", mod_name, 0.01, 0.01),
                        ])


    outputs = generate_paper_outputs(
        results=results,
        save_dir="/fshpc/jsiekier/promotion/CF_eval/paper_plots",
        decimals=5,
        show_plots=False,
        show_std=SHOW_STD_IN_PARETO,
        show_table_std=SHOW_STD_IN_TABLES,
        gamma_cmap="Blues",
    )
    '''
    beta_gamma = [(0.0, 0.0),
                  (0.0, 0.5), (0.0, 0.1), (0.0, 0.01), (0.0, 0.001),
                  (0.5, 0.5), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001),
                  (0.5, 0.0), (0.1, 0.0), (0.01, 0.0), (0.001, 0.0),
                  ]
    '''
    beta_gamma = [(0.0, 0.0),
                  (0.0,0.5),(0.0, 0.0001), (0.0, 0.1), (0.0, 0.01), (0.0, 0.001),
                  (0.5,0.5),(0.0001, 0.0001), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001),
                  (0.5,0.0),(0.0001, 0.0), (0.1, 0.0), (0.01, 0.0), (0.001, 0.0),
                  ]
    method_names = [('MLP', '0'), ('SPN', '1')]
    for mod_path, mod_name in spn_names:
        img_id = f'{ds}_largeSPN_{mod_name}'
        results=[]
        for m_name,f_str in method_names:
            for beta, gamma in beta_gamma:
                results.append((project_path + mod_path + grid_str + m_name+"b"+str(beta).replace('.','-')+
                                "g"+str(gamma).replace('.','-')+"-seg"+str(f_str)+"mean.json", m_name, mod_name,beta,gamma))

        outputs = generate_paper_outputs(
            results=results,
            save_dir="/fshpc/jsiekier/promotion/CF_eval/paper_plots",
            decimals=5,
            show_plots=False,
            show_std=SHOW_STD_IN_PARETO,
            show_table_std=SHOW_STD_IN_TABLES,
            gamma_cmap="Blues",
        )



    # -------------------------------------------------------------------------
    # Automated trade-off evaluation over all KLD weights, MLP/SPN, beta/gamma.
    # -------------------------------------------------------------------------
    all_grid_results = []
    for mod_path, mod_name in spn_names:
        for m_name, f_str in method_names:
            for beta, gamma in beta_gamma:
                all_grid_results.append((
                    project_path + mod_path + grid_str + m_name + "b" + str(beta).replace('.', '-') +
                    "g" + str(gamma).replace('.', '-') + "-seg" + str(f_str) + "mean.json",
                    m_name,
                    mod_name,
                    beta,
                    gamma,
                ))

    tradeoff_outputs = run_tradeoff_analysis(
        results=all_grid_results,
        save_dir="/fshpc/jsiekier/promotion/CF_eval/paper_plots",
        prefix=f"{ds}_largeSPN_tradeoffs",
        normalize_within_kld=True,
        top_k=3,
    )

    print("Trade-off best configs:")
    for row in tradeoff_outputs["best_configs"][:20]:
        print(row)

    # -------------------------------------------------------------------------
    # Additional integrated figures over all KLD weights.
    # These use the same all_grid_results as the automated trade-off analysis.
    # 1) 3x3 Pareto matrix: rows=KLD, columns=trade-off pairs.
    # 2) Appendix-style parameter sweep diagnostics for beta-only, gamma-only,
    #    and beta=gamma paths.
    # -------------------------------------------------------------------------
    all_grid_method_results = collect_method_results_paper(all_grid_results)

    plot_pareto_matrix_all_klds(
        all_grid_method_results,
        save_dir="/fshpc/jsiekier/promotion/CF_eval/paper_plots",
        show=False,
        show_std=SHOW_STD_IN_PARETO,
        gamma_cmap="Blues",
        gamma_log=True,
        beta_log=True,
        filename_prefix=f"{ds}_largeSPN_pareto_matrix_all_klds",
    )

    plot_parameter_tradeoff_sweeps_all_klds(
        all_grid_method_results,
        save_dir="/fshpc/jsiekier/promotion/CF_eval/paper_plots",
        show=False,
        filename_prefix=f"{ds}_largeSPN_parameter_tradeoff_sweeps_all_klds",
    )
