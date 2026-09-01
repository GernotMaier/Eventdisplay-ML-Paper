#!/usr/bin/env python3
"""Compare angular-containment radii stored in two ROOT histograms.

For every reconstructed-energy bin, the script normalizes the theta
distribution, integrates it outwards from zero, and determines the 68% and
95% containment radii. Quantiles are linearly interpolated within the theta
bin in which the cumulative distribution crosses the requested probability.

The accompanying panel compares the normalized theta distributions for events
with reconstructed energies above 1 TeV.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

DEFAULT_HISTOGRAM = "hthetaErec_DIFF"
MC_HISTOGRAM = "hthetaErec_SIMS"
CONTAINMENT_LEVELS = (0.68, 0.95)
MIN_DISTRIBUTION_ENERGY_TEV = 1.0
MAX_DISTRIBUTION_THETA_DEG = 0.2
DISTRIBUTION_REBIN_FACTOR = 5
DEFAULT_BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_PERCENTILES = (15.865, 84.135)

ContainmentResult = tuple[
    np.ndarray,
    dict[float, np.ndarray],
    dict[float, np.ndarray],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 68% and 95% angular-containment radii as a function of "
            "reconstructed energy for two ROOT files."
        )
    )
    parser.add_argument(
        "root_files",
        nargs=2,
        type=Path,
        metavar="ROOT_FILE",
        help="the two ROOT files to compare",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        metavar=("LABEL_1", "LABEL_2"),
        help="legend labels (default: parent directory names)",
    )
    parser.add_argument(
        "--histogram",
        default=DEFAULT_HISTOGRAM,
        help=f"ROOT histogram name (default: {DEFAULT_HISTOGRAM})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("angular_containment_comparison.pdf"),
        help="output plot; PDF is recommended (default: %(default)s)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="resolution for raster output (default: %(default)s)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="angular-containment y-axis range in degrees, e.g. --ylim 0 0.2",
    )
    parser.add_argument(
        "--plot_mc",
        action="store_true",
        help=(f"overlay containment curves and 1D distributions from {MC_HISTOGRAM}"),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="number of toys for containment uncertainties (default: %(default)s)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed for reproducible uncertainties (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.ylim is not None and args.ylim[0] >= args.ylim[1]:
        parser.error("--ylim requires YMIN < YMAX")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    return args


def _axis_title(axis: object) -> str:
    """Return a ROOT axis title without relying on uproot model internals."""
    try:
        return str(axis.member("fTitle"))
    except (AttributeError, KeyError):
        return ""


def containment_radii(
    root_file: Path,
    histogram_name: str = DEFAULT_HISTOGRAM,
    levels: tuple[float, ...] = CONTAINMENT_LEVELS,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 0,
) -> ContainmentResult:
    """Return energy centers, containment radii, and uncertainties in degrees.

    The expected histogram axes are reconstructed log10 energy on x and theta
    in degrees on y. Asymmetric uncertainties come from Gaussian toys using
    the ROOT bin variances. Underflow and overflow bins are excluded.
    """
    with uproot.open(root_file) as root:
        try:
            histogram = root[histogram_name]
        except uproot.KeyInFileError as error:
            raise KeyError(
                f"Histogram {histogram_name!r} not found in {root_file}"
            ) from error

        if len(histogram.axes) != 2:
            raise ValueError(
                f"{histogram_name!r} in {root_file} is not two-dimensional"
            )

        energy_edges = np.asarray(histogram.axes[0].edges(), dtype=float)
        theta_edges = np.asarray(histogram.axes[1].edges(), dtype=float)
        counts = np.asarray(histogram.values(flow=False), dtype=float)
        histogram_variances = histogram.variances(flow=False)
        axis_titles = tuple(_axis_title(axis) for axis in histogram.axes)

    if histogram_variances is None:
        raise ValueError(f"{histogram_name!r} in {root_file} has no bin variances")
    variances = np.asarray(histogram_variances, dtype=float)

    expected_shape = (energy_edges.size - 1, theta_edges.size - 1)
    if counts.shape != expected_shape:
        raise ValueError(
            f"Unexpected bin layout for {histogram_name!r} in {root_file}: "
            f"got {counts.shape}, expected {expected_shape}"
        )
    if variances.shape != expected_shape:
        raise ValueError(
            f"Unexpected variance layout for {histogram_name!r} in {root_file}: "
            f"got {variances.shape}, expected {expected_shape}"
        )
    if theta_edges[0] < 0 or not np.all(np.diff(theta_edges) > 0):
        raise ValueError("The theta axis must have increasing, non-negative edges")
    if not np.all(np.isfinite(counts)):
        raise ValueError("Histogram contains non-finite bin contents")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError("Histogram contains invalid bin variances")
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    if np.any(counts < 0):
        warnings.warn(
            f"Using signed bin contents from background-subtracted histogram "
            f"{histogram_name!r} in {root_file}",
            stacklevel=2,
        )

    if "log" not in axis_titles[0].lower() or "theta" not in axis_titles[1].lower():
        warnings.warn(
            f"Unexpected axis titles in {root_file}: {axis_titles!r}; "
            "assuming x=log10(reconstructed energy), y=theta.",
            stacklevel=2,
        )

    totals = counts.sum(axis=1)
    empty = totals <= 0
    if np.any(empty):
        warnings.warn(
            f"Ignoring {empty.sum()} empty energy bin(s) in {root_file}",
            stacklevel=2,
        )

    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    radii: dict[float, np.ndarray] = {}
    cumulative = np.cumsum(counts, axis=1)

    for level in levels:
        if not 0 < level < 1:
            raise ValueError(f"Containment level must be between zero and one: {level}")

        theta_quantiles = np.full(counts.shape[0], np.nan)
        for energy_bin in np.flatnonzero(~empty):
            target = level * totals[energy_bin]
            # A background-subtracted histogram can have negative tail bins,
            # making its cumulative sum non-monotonic. The first outward
            # crossing remains the physically relevant containment boundary.
            crossings = np.flatnonzero(cumulative[energy_bin] >= target)
            if crossings.size == 0:
                warnings.warn(
                    f"No {level:.0%} crossing in energy bin {energy_bin} of "
                    f"{root_file}",
                    stacklevel=2,
                )
                continue
            theta_bin = int(crossings[0])
            previous = cumulative[energy_bin, theta_bin - 1] if theta_bin else 0.0
            bin_count = counts[energy_bin, theta_bin]
            if bin_count <= 0:
                warnings.warn(
                    f"Invalid {level:.0%} crossing in energy bin {energy_bin} of "
                    f"{root_file}",
                    stacklevel=2,
                )
                continue
            fraction = np.clip((target - previous) / bin_count, 0.0, 1.0)
            bin_width = theta_edges[theta_bin + 1] - theta_edges[theta_bin]
            theta_quantiles[energy_bin] = theta_edges[theta_bin] + fraction * bin_width

        radii[level] = theta_quantiles

    radius_uncertainties = {
        level: np.full((2, counts.shape[0]), np.nan) for level in levels
    }
    random_generator = np.random.default_rng(random_seed)
    row_indices = np.arange(bootstrap_samples)

    for energy_bin in np.flatnonzero(~empty):
        toy_counts = random_generator.normal(
            loc=counts[energy_bin],
            scale=np.sqrt(variances[energy_bin]),
            size=(bootstrap_samples, counts.shape[1]),
        )
        toy_totals = toy_counts.sum(axis=1)
        valid_total = np.isfinite(toy_totals) & (toy_totals > 0)
        toy_cumulative = np.cumsum(toy_counts, axis=1)

        for level in levels:
            crossings = toy_cumulative >= level * toy_totals[:, np.newaxis]
            has_crossing = crossings.any(axis=1)
            crossing_bins = np.argmax(crossings, axis=1)
            previous_bins = np.maximum(crossing_bins - 1, 0)
            previous = toy_cumulative[row_indices, previous_bins]
            previous[crossing_bins == 0] = 0.0
            crossing_counts = toy_counts[row_indices, crossing_bins]
            valid_toy = valid_total & has_crossing & (crossing_counts > 0)

            if np.count_nonzero(valid_toy) < 100:
                warnings.warn(
                    f"Too few valid bootstrap toys for {level:.0%} containment "
                    f"in energy bin {energy_bin} of {root_file}",
                    stacklevel=2,
                )
                continue

            valid_bins = crossing_bins[valid_toy]
            fractions = np.clip(
                (level * toy_totals[valid_toy] - previous[valid_toy])
                / crossing_counts[valid_toy],
                0.0,
                1.0,
            )
            toy_quantiles = theta_edges[valid_bins] + fractions * (
                theta_edges[valid_bins + 1] - theta_edges[valid_bins]
            )
            lower, upper = np.percentile(toy_quantiles, BOOTSTRAP_PERCENTILES)
            central = radii[level][energy_bin]
            radius_uncertainties[level][:, energy_bin] = (
                max(central - lower, 0.0),
                max(upper - central, 0.0),
            )

    return energy_centers, radii, radius_uncertainties


def angular_distribution(
    root_file: Path,
    histogram_name: str = DEFAULT_HISTOGRAM,
    minimum_energy_tev: float = MIN_DISTRIBUTION_ENERGY_TEV,
    rebin_factor: int = DISTRIBUTION_REBIN_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a rebinned theta density above a reconstructed-energy threshold.

    The bin containing the threshold and all higher-energy bins are projected
    onto the theta axis. Adjacent theta bins are then summed before normalizing
    the distribution.
    """
    with uproot.open(root_file) as root:
        try:
            histogram = root[histogram_name]
        except uproot.KeyInFileError as error:
            raise KeyError(
                f"Histogram {histogram_name!r} not found in {root_file}"
            ) from error

        if len(histogram.axes) != 2:
            raise ValueError(
                f"{histogram_name!r} in {root_file} is not two-dimensional"
            )

        energy_edges = np.asarray(histogram.axes[0].edges(), dtype=float)
        theta_edges = np.asarray(histogram.axes[1].edges(), dtype=float)
        counts = np.asarray(histogram.values(flow=False), dtype=float)

    target_log_energy = np.log10(minimum_energy_tev)
    if not energy_edges[0] <= target_log_energy < energy_edges[-1]:
        raise ValueError(
            f"{minimum_energy_tev:g} TeV is outside the energy range in {root_file}"
        )

    first_energy_bin = (
        np.searchsorted(energy_edges, target_log_energy, side="right") - 1
    )
    theta_counts = counts[first_energy_bin:].sum(axis=0)
    if rebin_factor < 1:
        raise ValueError("The distribution rebin factor must be positive")
    if theta_counts.size % rebin_factor:
        raise ValueError(
            f"Cannot rebin {theta_counts.size} theta bins by {rebin_factor}"
        )

    theta_counts = theta_counts.reshape(-1, rebin_factor).sum(axis=1)
    theta_edges = theta_edges[::rebin_factor]
    total = theta_counts.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"{histogram_name!r} in {root_file} has no positive total above "
            f"{minimum_energy_tev:g} TeV"
        )

    # Dividing each bin's probability by Delta theta gives a density whose
    # integral is one, while retaining signed background-subtracted bins.
    theta_density = theta_counts / total / np.diff(theta_edges)
    return theta_edges, theta_density


def default_label(path: Path) -> str:
    """Derive a useful label even when both files have the same basename."""
    return path.parent.name if path.parent.name else path.stem


def make_plot(
    datasets: list[ContainmentResult],
    distributions: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    ylim: tuple[float, float] | None = None,
    mc_datasets: list[ContainmentResult] | None = None,
    mc_distributions: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Create a compact, publication-ready comparison figure."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (axis, distribution_axis) = plt.subplots(
        ncols=2,
        figsize=(10.0, 4.4),
        gridspec_kw={"width_ratios": (1.65, 1.0)},
        constrained_layout=True,
    )
    colors = ("#0072B2", "#D55E00")
    styles = {0.68: ("o", "-"), 0.95: ("s", "--")}
    mc_styles = {0.68: ("o", ":"), 0.95: ("s", "-.")}

    for (energy, radii, errors), label, color in zip(
        datasets, labels, colors, strict=True
    ):
        for level in CONTAINMENT_LEVELS:
            marker, line_style = styles[level]
            valid = np.isfinite(radii[level]) & np.all(
                np.isfinite(errors[level]), axis=0
            )
            axis.errorbar(
                energy[valid],
                radii[level][valid],
                yerr=errors[level][:, valid],
                marker=marker,
                linestyle=line_style,
                color=color,
                linewidth=1.6,
                markersize=5.0,
                markerfacecolor="white",
                markeredgewidth=1.2,
                elinewidth=0.55,
                capsize=1.5,
                capthick=0.55,
                label=f"{label} ({level:.0%})",
            )

    if mc_datasets is not None:
        for (energy, radii, errors), label, color in zip(
            mc_datasets, labels, colors, strict=True
        ):
            for level in CONTAINMENT_LEVELS:
                marker, line_style = mc_styles[level]
                valid = np.isfinite(radii[level]) & np.all(
                    np.isfinite(errors[level]), axis=0
                )
                axis.errorbar(
                    energy[valid],
                    radii[level][valid],
                    yerr=errors[level][:, valid],
                    marker=marker,
                    linestyle=line_style,
                    color=color,
                    linewidth=1.4,
                    markersize=4.5,
                    markerfacecolor=color,
                    markeredgewidth=1.0,
                    elinewidth=0.55,
                    capsize=1.5,
                    capthick=0.55,
                    label=f"{label} (MC, {level:.0%})",
                )

    axis.set_xlabel(r"$\log_{10}(E_{\mathrm{rec}}\,/\,\mathrm{TeV})$")
    axis.set_ylabel("Angular containment radius [deg]")
    energy_min = min(data[0][0] for data in datasets)
    energy_max = max(data[0][-1] for data in datasets)
    energy_padding = 0.03 * (energy_max - energy_min)
    axis.set_xlim(energy_min - energy_padding, energy_max + energy_padding)
    axis.set_ylim(*(ylim if ylim is not None else (0, None)))
    axis.grid(which="major", color="0.88", linewidth=0.6)
    axis.legend(
        frameon=False,
        ncol=2,
        columnspacing=1.0,
        handlelength=2.4,
        fontsize=7.5 if mc_datasets is not None else 9,
    )

    for (theta_edges, theta_density), label, color in zip(
        distributions, labels, colors, strict=True
    ):
        distribution_axis.stairs(
            theta_density,
            theta_edges,
            color=color,
            linewidth=1.6,
            label=label,
        )

    if mc_distributions is not None:
        for (theta_edges, theta_density), label, color in zip(
            mc_distributions, labels, colors, strict=True
        ):
            distribution_axis.stairs(
                theta_density,
                theta_edges,
                color=color,
                linestyle="--",
                linewidth=1.6,
                label=f"{label} (MC)",
            )

    distribution_axis.axhline(0, color="0.35", linewidth=0.7)
    distribution_axis.set_xlim(0, MAX_DISTRIBUTION_THETA_DEG)
    distribution_axis.set_xlabel(r"Angular separation $\theta$ [deg]")
    distribution_axis.set_ylabel(r"Normalized density [deg$^{-1}$]")
    distribution_axis.set_title(r"$E_{\mathrm{rec}} > 1\,\mathrm{TeV}$")
    distribution_axis.grid(which="major", color="0.88", linewidth=0.6)
    distribution_axis.legend(frameon=False)
    return figure, (axis, distribution_axis)


def main() -> None:
    args = parse_args()
    labels = (
        list(args.labels)
        if args.labels
        else [default_label(path) for path in args.root_files]
    )
    datasets = [
        containment_radii(
            root_file,
            args.histogram,
            bootstrap_samples=args.bootstrap_samples,
            random_seed=args.random_seed + file_index,
        )
        for file_index, root_file in enumerate(args.root_files)
    ]
    distributions = [
        angular_distribution(root_file, args.histogram) for root_file in args.root_files
    ]
    mc_distributions = (
        [angular_distribution(root_file, MC_HISTOGRAM) for root_file in args.root_files]
        if args.plot_mc
        else None
    )
    mc_datasets = (
        [
            containment_radii(
                root_file,
                MC_HISTOGRAM,
                bootstrap_samples=args.bootstrap_samples,
                random_seed=(args.random_seed + len(args.root_files) + file_index),
            )
            for file_index, root_file in enumerate(args.root_files)
        ]
        if args.plot_mc
        else None
    )
    ylim = tuple(args.ylim) if args.ylim is not None else None
    figure, _ = make_plot(
        datasets,
        distributions,
        labels,
        ylim=ylim,
        mc_datasets=mc_datasets,
        mc_distributions=mc_distributions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
