#!/usr/bin/env python3
"""Compare angular containment with a shape-constrained ON/OFF likelihood.

The signal in each theta annulus is fitted from the raw ON and OFF counts
using a joint Poisson likelihood. Rather than imposing a PSF function, the
only shape assumption is that the signal surface brightness does not increase
with angular separation. Containment radii are read from the resulting
non-parametric cumulative signal profile.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

DEFAULT_ON_HISTOGRAM = "hthetaErec_ON"
DEFAULT_OFF_HISTOGRAM = "hthetaErec_OFF"
DEFAULT_MC_HISTOGRAM = "hthetaErec_SIMS"
DEFAULT_ALPHA = 0.2
CONTAINMENT_LEVELS = (0.68, 0.95)
REFERENCE_ENERGY_TEV = 1.0
THETA_REBIN_FACTOR = 10
DEFAULT_BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_PERCENTILES = (15.865, 84.135)
MAX_DISTRIBUTION_THETA_DEG = 0.4


@dataclass(frozen=True)
class ContainmentFit:
    energy_centers: np.ndarray
    radii: dict[float, np.ndarray]
    uncertainties: dict[float, np.ndarray]


@dataclass(frozen=True)
class DistributionProfile:
    theta_edges: np.ndarray
    observed_density: np.ndarray
    observed_uncertainty: np.ndarray
    fitted_density: np.ndarray


@dataclass(frozen=True)
class OnOffProfile:
    signal: np.ndarray
    background: np.ndarray
    significance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare angular-containment radii using a shape-constrained "
            "ON/OFF likelihood."
        )
    )
    parser.add_argument("root_files", nargs=2, type=Path, metavar="ROOT_FILE")
    parser.add_argument("--labels", nargs=2, metavar=("LABEL_1", "LABEL_2"))
    parser.add_argument("--on-histogram", default=DEFAULT_ON_HISTOGRAM)
    parser.add_argument("--off-histogram", default=DEFAULT_OFF_HISTOGRAM)
    parser.add_argument("--mc-histogram", default=DEFAULT_MC_HISTOGRAM)
    parser.add_argument(
        "--plot_mc",
        "--plot-mc",
        dest="plot_mc",
        action="store_true",
        help="overlay empirical MC containment from the SIMS histogram",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="ON/OFF exposure ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--theta-rebin",
        type=int,
        default=THETA_REBIN_FACTOR,
        help="number of original theta bins per likelihood annulus (default: %(default)s)",
    )
    parser.add_argument(
        "--fit-theta-max",
        type=float,
        help="maximum theta used for containment and profile fitting [deg]",
    )
    parser.add_argument(
        "--min-counts",
        type=float,
        default=20.0,
        help="minimum ON+OFF entries in an energy bin (default: %(default)s)",
    )
    parser.add_argument(
        "--min-significance",
        type=float,
        default=2.0,
        help="minimum likelihood signal significance in an energy bin (default: %(default)s)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="number of ON/OFF Poisson toys for intervals (default: %(default)s)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed for reproducible intervals (default: %(default)s)",
    )
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"))
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("angular_containment_onoff_comparison.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be positive")
    if args.theta_rebin < 1:
        parser.error("--theta-rebin must be positive")
    if args.fit_theta_max is not None and args.fit_theta_max <= 0:
        parser.error("--fit-theta-max must be positive")
    if args.min_counts < 0 or args.min_significance < 0:
        parser.error("minimum counts and significance must be non-negative")
    if args.bootstrap_samples < 20:
        parser.error("--bootstrap-samples must be at least 20")
    if args.ylim is not None and args.ylim[0] >= args.ylim[1]:
        parser.error("--ylim requires YMIN < YMAX")
    return args


def _axis_title(axis: object) -> str:
    try:
        return str(axis.member("fTitle"))
    except (AttributeError, KeyError):
        return ""


def _read_histogram(
    root: uproot.ReadOnlyDirectory, name: str, root_file: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, str]]:
    try:
        histogram = root[name]
    except uproot.KeyInFileError as error:
        candidates = [
            str(key).split(";")[0] for key in root.keys() if "theta" in str(key).lower()
        ]
        raise KeyError(
            f"Histogram {name!r} not found in {root_file}. "
            f"Available theta histograms: {candidates}"
        ) from error
    if len(histogram.axes) != 2:
        raise ValueError(f"{name!r} in {root_file} is not two-dimensional")
    return (
        np.asarray(histogram.axes[0].edges(), dtype=float),
        np.asarray(histogram.axes[1].edges(), dtype=float),
        np.asarray(histogram.values(flow=False), dtype=float),
        tuple(_axis_title(axis) for axis in histogram.axes),
    )


def read_on_off(
    root_file: Path, on_name: str, off_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with uproot.open(root_file) as root:
        energy_edges, theta_edges, on, titles = _read_histogram(
            root, on_name, root_file
        )
        off_energy, off_theta, off, _ = _read_histogram(root, off_name, root_file)
    expected_shape = (energy_edges.size - 1, theta_edges.size - 1)
    if (
        on.shape != expected_shape
        or off.shape != expected_shape
        or not np.array_equal(energy_edges, off_energy)
        or not np.array_equal(theta_edges, off_theta)
    ):
        raise ValueError(f"ON/OFF histogram binning does not match in {root_file}")
    if not np.all(np.isfinite(on)) or not np.all(np.isfinite(off)):
        raise ValueError("ON/OFF histograms contain non-finite values")
    if np.any(on < 0) or np.any(off < 0):
        raise ValueError("ON/OFF histograms must contain non-negative counts")
    if theta_edges[0] < 0 or not np.all(np.diff(theta_edges) > 0):
        raise ValueError("theta edges must be increasing and non-negative")
    if "2" in titles[1] or "squared" in titles[1].lower():
        raise ValueError("This program expects theta, not theta-squared")
    return energy_edges, theta_edges, on, off


def read_mc(
    root_file: Path, histogram_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with uproot.open(root_file) as root:
        energy_edges, theta_edges, counts, titles = _read_histogram(
            root, histogram_name, root_file
        )
    expected_shape = (energy_edges.size - 1, theta_edges.size - 1)
    if counts.shape != expected_shape or not np.all(np.isfinite(counts)):
        raise ValueError(f"Invalid MC histogram {histogram_name!r} in {root_file}")
    if np.any(counts < 0):
        raise ValueError("MC histogram contains negative weights")
    if "2" in titles[1] or "squared" in titles[1].lower():
        raise ValueError("This program expects theta, not theta-squared")
    return energy_edges, theta_edges, counts


def _truncate_and_rebin(
    theta_edges: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    rebin_factor: int,
    theta_max: float | None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if theta_max is not None:
        bins = np.searchsorted(theta_edges, theta_max, side="right") - 1
        if bins < 2:
            raise ValueError("--fit-theta-max leaves fewer than two theta bins")
        theta_edges = theta_edges[: bins + 1]
        arrays = tuple(array[..., :bins] for array in arrays)
    bins = theta_edges.size - 1
    if bins % rebin_factor:
        raise ValueError(
            f"Cannot rebin {bins} theta bins by {rebin_factor}; choose a divisor"
        )
    rebinned = tuple(
        array.reshape(*array.shape[:-1], -1, rebin_factor).sum(axis=-1)
        for array in arrays
    )
    return theta_edges[::rebin_factor], rebinned


def _profile_background(
    signal: np.ndarray, on: np.ndarray, off: np.ndarray, alpha: float
) -> np.ndarray:
    quadratic = alpha * (1.0 + alpha)
    linear = (1.0 + alpha) * signal - alpha * (on + off)
    discriminant = linear**2 + 4.0 * quadratic * off * signal
    return (-linear + np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * quadratic)


def _poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
    positive = observed > 0
    if np.any(expected[positive] <= 0):
        return np.inf
    return float(
        expected.sum() - np.sum(observed[positive] * np.log(expected[positive]))
    )


def _annular_nll(
    surface_brightness: float,
    areas: np.ndarray,
    on: np.ndarray,
    off: np.ndarray,
    alpha: float,
) -> float:
    signal = surface_brightness * areas
    background = _profile_background(signal, on, off, alpha)
    return _poisson_nll(on, signal + alpha * background) + _poisson_nll(off, background)


def _block_minimum(
    areas: np.ndarray, on: np.ndarray, off: np.ndarray, alpha: float
) -> float:
    """Minimize one convex pooled-annulus likelihood with golden search."""
    density_scale = max((on.sum() + alpha * off.sum()) / areas.sum(), 1.0)
    lower = 0.0
    upper = density_scale
    objective = lambda value: _annular_nll(value, areas, on, off, alpha)
    minimum = objective(lower)
    while objective(upper) < minimum and upper < density_scale * 1e6:
        minimum = objective(upper)
        upper *= 2.0
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = lower + (1.0 - ratio) * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value = objective(left)
    right_value = objective(right)
    for _ in range(48):
        if left_value <= right_value:
            upper, right, right_value = right, left, left_value
            left = lower + (1.0 - ratio) * (upper - lower)
            left_value = objective(left)
        else:
            lower, left, left_value = left, right, right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
    return 0.5 * (lower + upper)


def fit_on_off_profile(
    theta_edges: np.ndarray, on: np.ndarray, off: np.ndarray, alpha: float
) -> OnOffProfile:
    """Non-parametric ON/OFF MLE with non-increasing surface brightness.

    A generalized pool-adjacent-violators algorithm finds the likelihood MLE
    under the physical radial-shape constraint.
    """
    areas = np.pi * np.diff(theta_edges**2)
    blocks: list[tuple[int, int, float]] = []
    for index in range(on.size):
        blocks.append(
            (
                index,
                index + 1,
                _block_minimum(
                    areas[index : index + 1],
                    on[index : index + 1],
                    off[index : index + 1],
                    alpha,
                ),
            )
        )
        while len(blocks) > 1 and blocks[-2][2] < blocks[-1][2]:
            start = blocks[-2][0]
            end = blocks[-1][1]
            blocks[-2:] = [
                (
                    start,
                    end,
                    _block_minimum(
                        areas[start:end], on[start:end], off[start:end], alpha
                    ),
                )
            ]

    surface_brightness = np.empty(on.size)
    for start, end, value in blocks:
        surface_brightness[start:end] = value
    signal = surface_brightness * areas
    background = _profile_background(signal, on, off, alpha)
    null_background = _profile_background(np.zeros_like(signal), on, off, alpha)
    null_nll = _poisson_nll(on, alpha * null_background) + _poisson_nll(
        off, null_background
    )
    fitted_nll = _poisson_nll(on, signal + alpha * background) + _poisson_nll(
        off, background
    )
    significance = np.sqrt(max(2.0 * (null_nll - fitted_nll), 0.0))
    return OnOffProfile(signal, background, significance)


def _containment_radius(
    theta_edges: np.ndarray, signal: np.ndarray, level: float
) -> float:
    total = signal.sum()
    if not np.isfinite(total) or total <= 0:
        return np.nan
    cumulative = np.cumsum(signal)
    index = min(int(np.searchsorted(cumulative, level * total)), signal.size - 1)
    previous = cumulative[index - 1] if index else 0.0
    fraction = np.clip((level * total - previous) / signal[index], 0.0, 1.0)
    return float(
        np.sqrt(
            theta_edges[index] ** 2
            + fraction * (theta_edges[index + 1] ** 2 - theta_edges[index] ** 2)
        )
    )


def _empirical_radius(
    theta_edges: np.ndarray, counts: np.ndarray, level: float
) -> float:
    total = counts.sum()
    if total <= 0:
        return np.nan
    cumulative = np.cumsum(counts)
    index = min(int(np.searchsorted(cumulative, level * total)), counts.size - 1)
    previous = cumulative[index - 1] if index else 0.0
    fraction = np.clip((level * total - previous) / counts[index], 0.0, 1.0)
    return float(
        theta_edges[index] + fraction * (theta_edges[index + 1] - theta_edges[index])
    )


def _bootstrap_errors(
    theta_edges: np.ndarray,
    profile: OnOffProfile,
    alpha: float,
    samples: int,
    rng: np.random.Generator,
) -> dict[float, tuple[float, float]]:
    radii = {level: [] for level in CONTAINMENT_LEVELS}
    expected_on = profile.signal + alpha * profile.background
    for _ in range(samples):
        toy_on = rng.poisson(expected_on)
        toy_off = rng.poisson(profile.background)
        toy_profile = fit_on_off_profile(theta_edges, toy_on, toy_off, alpha)
        for level in CONTAINMENT_LEVELS:
            radius = _containment_radius(theta_edges, toy_profile.signal, level)
            if np.isfinite(radius):
                radii[level].append(radius)
    errors: dict[float, tuple[float, float]] = {}
    for level, values in radii.items():
        if len(values) < 20:
            errors[level] = (np.nan, np.nan)
            continue
        lower, upper = np.percentile(values, BOOTSTRAP_PERCENTILES)
        central = _containment_radius(theta_edges, profile.signal, level)
        errors[level] = (max(central - lower, 0.0), max(upper - central, 0.0))
    return errors


def containment_from_on_off(
    root_file: Path,
    on_name: str,
    off_name: str,
    alpha: float,
    rebin_factor: int,
    theta_max: float | None,
    min_counts: float,
    min_significance: float,
    bootstrap_samples: int,
    random_seed: int,
) -> ContainmentFit:
    energy_edges, theta_edges, on, off = read_on_off(root_file, on_name, off_name)
    theta_edges, (on, off) = _truncate_and_rebin(
        theta_edges, (on, off), rebin_factor, theta_max
    )
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    radii = {
        level: np.full(energy_centers.shape, np.nan) for level in CONTAINMENT_LEVELS
    }
    errors = {
        level: np.full((2, energy_centers.size), np.nan) for level in CONTAINMENT_LEVELS
    }
    rng = np.random.default_rng(random_seed)
    for index, energy in enumerate(energy_centers):
        if on[index].sum() + off[index].sum() < min_counts:
            continue
        profile = fit_on_off_profile(theta_edges, on[index], off[index], alpha)
        if profile.significance < min_significance:
            warnings.warn(
                f"Skipping log10(E/TeV)={energy:.3f} in {root_file}: "
                f"signal significance is {profile.significance:.2f} sigma",
                stacklevel=2,
            )
            continue
        toy_errors = _bootstrap_errors(
            theta_edges, profile, alpha, bootstrap_samples, rng
        )
        for level in CONTAINMENT_LEVELS:
            radii[level][index] = _containment_radius(
                theta_edges, profile.signal, level
            )
            errors[level][:, index] = toy_errors[level]
    return ContainmentFit(energy_centers, radii, errors)


def containment_from_mc(
    root_file: Path,
    histogram_name: str,
    rebin_factor: int,
    theta_max: float | None,
) -> ContainmentFit:
    energy_edges, theta_edges, counts = read_mc(root_file, histogram_name)
    theta_edges, (counts,) = _truncate_and_rebin(
        theta_edges, (counts,), rebin_factor, theta_max
    )
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    radii = {
        level: np.full(energy_centers.shape, np.nan) for level in CONTAINMENT_LEVELS
    }
    errors = {
        level: np.full((2, energy_centers.size), np.nan) for level in CONTAINMENT_LEVELS
    }
    for index, distribution in enumerate(counts):
        for level in CONTAINMENT_LEVELS:
            radii[level][index] = _empirical_radius(theta_edges, distribution, level)
    return ContainmentFit(energy_centers, radii, errors)


def _one_tev_bin(energy_edges: np.ndarray) -> int:
    target = np.log10(REFERENCE_ENERGY_TEV)
    if not energy_edges[0] <= target < energy_edges[-1]:
        raise ValueError("1 TeV is outside the histogram energy range")
    return int(np.searchsorted(energy_edges, target, side="right") - 1)


def profile_from_on_off(
    root_file: Path,
    on_name: str,
    off_name: str,
    alpha: float,
    rebin_factor: int,
    theta_max: float | None,
) -> DistributionProfile:
    energy_edges, theta_edges, on, off = read_on_off(root_file, on_name, off_name)
    theta_edges, (on, off) = _truncate_and_rebin(
        theta_edges, (on, off), rebin_factor, theta_max
    )
    index = _one_tev_bin(energy_edges)
    profile = fit_on_off_profile(theta_edges, on[index], off[index], alpha)
    excess = on[index] - alpha * off[index]
    total = excess.sum()
    if total <= 0:
        raise ValueError(f"1 TeV excess is not positive in {root_file}")
    widths = np.diff(theta_edges)
    return DistributionProfile(
        theta_edges,
        excess / total / widths,
        np.sqrt(on[index] + alpha**2 * off[index]) / total / widths,
        profile.signal / profile.signal.sum() / widths,
    )


def profile_from_mc(
    root_file: Path,
    histogram_name: str,
    rebin_factor: int,
    theta_max: float | None,
) -> DistributionProfile:
    energy_edges, theta_edges, counts = read_mc(root_file, histogram_name)
    theta_edges, (counts,) = _truncate_and_rebin(
        theta_edges, (counts,), rebin_factor, theta_max
    )
    distribution = counts[_one_tev_bin(energy_edges)]
    widths = np.diff(theta_edges)
    total = distribution.sum()
    return DistributionProfile(
        theta_edges,
        distribution / total / widths,
        np.sqrt(distribution) / total / widths,
        distribution / total / widths,
    )


def default_label(path: Path) -> str:
    return path.parent.name if path.parent.name else path.stem


def make_plot(
    data: list[ContainmentFit],
    profiles: list[DistributionProfile],
    labels: list[str],
    ylim: tuple[float, float] | None,
    mc_data: list[ContainmentFit] | None,
    mc_profiles: list[DistributionProfile] | None,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (axis, profile_axis) = plt.subplots(
        ncols=2,
        figsize=(10.5, 4.5),
        gridspec_kw={"width_ratios": (1.6, 1.0)},
        constrained_layout=True,
    )
    colors = ("#0072B2", "#D55E00")
    styles = {0.68: ("o", "-"), 0.95: ("s", "--")}
    mc_styles = {0.68: ("o", ":"), 0.95: ("s", "-.")}
    for fit, label, color in zip(data, labels, colors, strict=True):
        for level in CONTAINMENT_LEVELS:
            marker, line = styles[level]
            valid = np.isfinite(fit.radii[level])
            axis.errorbar(
                fit.energy_centers[valid],
                fit.radii[level][valid],
                yerr=fit.uncertainties[level][:, valid],
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.6,
                markersize=5,
                markerfacecolor="white",
                capsize=1.5,
                elinewidth=0.6,
                label=f"{label} ({level:.0%})",
            )
    if mc_data is not None:
        for fit, label, color in zip(mc_data, labels, colors, strict=True):
            for level in CONTAINMENT_LEVELS:
                marker, line = mc_styles[level]
                valid = np.isfinite(fit.radii[level])
                axis.plot(
                    fit.energy_centers[valid],
                    fit.radii[level][valid],
                    color=color,
                    marker=marker,
                    linestyle=line,
                    linewidth=1.4,
                    markersize=4.5,
                    label=f"{label} (MC, {level:.0%})",
                )
    energy_min = min(fit.energy_centers[0] for fit in data)
    energy_max = max(fit.energy_centers[-1] for fit in data)
    axis.set_xlim(
        energy_min - 0.03 * (energy_max - energy_min),
        energy_max + 0.03 * (energy_max - energy_min),
    )
    axis.set_ylim(*(ylim if ylim is not None else (0, None)))
    axis.set_xlabel(r"$\log_{10}(E_{\mathrm{rec}}/\mathrm{TeV})$")
    axis.set_ylabel("Angular containment radius [deg]")
    axis.grid(color="0.88", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, fontsize=7.5 if mc_data is not None else 9)

    for profile, label, color in zip(profiles, labels, colors, strict=True):
        centers = 0.5 * (profile.theta_edges[:-1] + profile.theta_edges[1:])
        profile_axis.errorbar(
            centers,
            profile.observed_density,
            yerr=profile.observed_uncertainty,
            color=color,
            marker="o",
            linestyle="none",
            markersize=3.5,
            capsize=1.2,
            elinewidth=0.6,
            label=f"{label}: ON$-\\alpha$OFF",
        )
        profile_axis.stairs(
            profile.fitted_density,
            profile.theta_edges,
            color=color,
            linewidth=1.7,
            label=f"{label}: likelihood profile",
        )
    if mc_profiles is not None:
        for profile, label, color in zip(mc_profiles, labels, colors, strict=True):
            profile_axis.stairs(
                profile.observed_density,
                profile.theta_edges,
                color=color,
                linestyle="--",
                linewidth=1.4,
                label=f"{label}: MC",
            )
    profile_axis.axhline(0, color="0.35", linewidth=0.7)
    profile_axis.set_xlim(
        0,
        min(
            MAX_DISTRIBUTION_THETA_DEG,
            max(profile.theta_edges[-1] for profile in profiles),
        ),
    )
    profile_axis.set_xlabel(r"Angular separation $\theta$ [deg]")
    profile_axis.set_ylabel(r"Normalized density [deg$^{-1}$]")
    profile_axis.set_title(r"$E_{\mathrm{rec}} = 1\,\mathrm{TeV}$ bin")
    profile_axis.grid(color="0.88", linewidth=0.6)
    profile_axis.legend(
        frameon=False, ncol=2, fontsize=6.2, columnspacing=0.8, handlelength=2.0
    )
    return figure


def main() -> None:
    args = parse_args()
    labels = (
        list(args.labels)
        if args.labels
        else [default_label(path) for path in args.root_files]
    )
    data = [
        containment_from_on_off(
            path,
            args.on_histogram,
            args.off_histogram,
            args.alpha,
            args.theta_rebin,
            args.fit_theta_max,
            args.min_counts,
            args.min_significance,
            args.bootstrap_samples,
            args.random_seed + index,
        )
        for index, path in enumerate(args.root_files)
    ]
    profiles = [
        profile_from_on_off(
            path,
            args.on_histogram,
            args.off_histogram,
            args.alpha,
            args.theta_rebin,
            args.fit_theta_max,
        )
        for path in args.root_files
    ]
    mc_data = (
        [
            containment_from_mc(
                path, args.mc_histogram, args.theta_rebin, args.fit_theta_max
            )
            for path in args.root_files
        ]
        if args.plot_mc
        else None
    )
    mc_profiles = (
        [
            profile_from_mc(
                path, args.mc_histogram, args.theta_rebin, args.fit_theta_max
            )
            for path in args.root_files
        ]
        if args.plot_mc
        else None
    )
    figure = make_plot(
        data,
        profiles,
        labels,
        tuple(args.ylim) if args.ylim else None,
        mc_data,
        mc_profiles,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
