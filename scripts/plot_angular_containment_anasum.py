#!/usr/bin/python3
"""Compare anasum angular containment with histogram-derived MC resolution.

Data are read from the background-subtracted ``htheta2Erec_diff`` histogram
written by anasum.  MC curves are derived from the reconstructed-energy
``hAngularLogDiff_2D`` histogram stored in the IRF tree.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.optimize import OptimizeWarning, curve_fit

DEFAULT_DATA_HISTOGRAM = "total_1/stereo/stereoParameterHistograms/htheta2Erec_diff"
DEFAULT_MC_TREE = "t_angular_resolution"
MC_LOGDIFF_HISTOGRAM_INDEX = 2  # E_LOGDIFF, hAngularLogDiff_2D
DEFAULT_MAX_THETA_DEG = 0.5
DEFAULT_CONTAINMENT = 0.68
DEFAULT_BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_PERCENTILES = (15.865, 84.135)
DIAGNOSTIC_ENERGY_CENTERS = (-0.5, 0.0, 0.5, 1.0)
DIAGNOSTIC_MAX_THETA_DEG = 0.25
DEFAULT_DISTRIBUTION_OUTPUT = Path("angular_distributions_anasum_comparison.pdf")


@dataclass(frozen=True)
class ContainmentResult:
    energy_centers: np.ndarray
    radii: np.ndarray
    uncertainties: np.ndarray


@dataclass(frozen=True)
class DataHistogram:
    energy_edges: np.ndarray
    theta_edges: np.ndarray
    counts: np.ndarray
    variances: np.ndarray
    theta_squared: bool = True


def _parse_containment(value: str) -> float:
    """Parse a containment fraction, accepting either 0.68 or 68 notation."""
    try:
        containment = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("containment must be a number") from error
    if containment > 1:
        containment /= 100.0
    if not 0 < containment < 1:
        raise argparse.ArgumentTypeError("containment must be between 0 and 1")
    percentage = 100 * containment
    if not np.isclose(percentage, round(percentage)):
        raise argparse.ArgumentTypeError(
            "containment must correspond to an integer percentage so the MC tree "
            "name can be selected"
        )
    return containment


def _mc_tree_for_containment(containment: float) -> str:
    """Return the standard angular-resolution tree for a containment level."""
    percentage = int(round(100 * containment))
    if percentage == 68:
        return DEFAULT_MC_TREE
    return f"{DEFAULT_MC_TREE}_{percentage:03d}p"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one configurable angular-containment radius from one or more "
            "anasum files with curves derived from the corresponding IRF objects."
        )
    )
    parser.add_argument(
        "data_files",
        nargs="+",
        type=Path,
        metavar="ANASUM_FILE",
        help="one or more anasum output files to compare",
    )
    parser.add_argument(
        "--mc-files",
        nargs="+",
        type=Path,
        required=True,
        metavar="IRF_FILE",
        help="matching IRF files containing the angular-resolution histogram",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="one label per file (default: parent directory names)",
    )
    parser.add_argument(
        "--data-histogram",
        default=DEFAULT_DATA_HISTOGRAM,
        help=f"anasum histogram (default: {DEFAULT_DATA_HISTOGRAM})",
    )
    theta_limit_group = parser.add_mutually_exclusive_group()
    theta_limit_group.add_argument(
        "--max-theta",
        type=float,
        default=DEFAULT_MAX_THETA_DEG,
        help="maximum angular separation for fits and selected-bin plot [deg] (default: 0.5)",
    )
    theta_limit_group.add_argument(
        "--max-theta2",
        type=float,
        help="maximum theta squared for fits and selected-bin plot [deg^2]",
    )
    parser.add_argument(
        "--method",
        choices=("double-gaussian", "king", "cumulative"),
        default="double-gaussian",
        help="data containment estimator (default: %(default)s)",
    )
    parser.add_argument(
        "--containment",
        type=_parse_containment,
        default=DEFAULT_CONTAINMENT,
        metavar="FRACTION",
        help=(
            "containment level as a fraction or percentage (default: 0.68; "
            "for example, --containment 0.95 or --containment 95)"
        ),
    )
    parser.add_argument(
        "--mc-tree",
        help=(
            "override the IRF tree containing hAngularLogDiff_2D "
            "(default: selected from --containment)"
        ),
    )
    parser.add_argument(
        "--mc-entry",
        type=int,
        default=0,
        help="IRF tree entry (azimuth bin) to use (default: %(default)s)",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="number of Gaussian toys for data intervals (default: %(default)s)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed for reproducible data intervals (default: %(default)s)",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="angular-containment y-axis range in degrees",
    )
    parser.add_argument(
        "--energy-range",
        "--xlim",
        dest="energy_range",
        nargs=2,
        type=float,
        metavar=("EMIN", "EMAX"),
        help="containment x-axis range in log10(Erec/TeV)",
    )
    parser.add_argument(
        "--energy-rebin",
        type=int,
        default=1,
        metavar="FACTOR",
        help="combine this many adjacent data energy bins (default: 1; MC is unchanged)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("angular_containment_anasum_comparison.pdf"),
        help="output plot (default: %(default)s)",
    )
    parser.add_argument(
        "--distribution-output",
        type=Path,
        default=DEFAULT_DISTRIBUTION_OUTPUT,
        help=(
            "output PDF for the four angular-distribution diagnostics "
            f"(default: {DEFAULT_DISTRIBUTION_OUTPUT})"
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.max_theta is not None and (
        not np.isfinite(args.max_theta) or args.max_theta <= 0
    ):
        parser.error("--max-theta must be positive")
    if args.max_theta2 is not None and (
        not np.isfinite(args.max_theta2) or args.max_theta2 <= 0
    ):
        parser.error("--max-theta2 must be positive")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.ylim is not None and args.ylim[0] >= args.ylim[1]:
        parser.error("--ylim requires YMIN < YMAX")
    if args.energy_range is not None and (
        not np.all(np.isfinite(args.energy_range))
        or args.energy_range[0] >= args.energy_range[1]
    ):
        parser.error("--energy-range requires finite EMIN < EMAX")
    if args.energy_rebin < 1:
        parser.error("--energy-rebin must be at least 1")
    if len(args.data_files) != len(args.mc_files):
        parser.error(
            "the number of data files and MC files must be the same "
            "(one file each is supported)"
        )
    if args.labels is not None and len(args.labels) != len(args.data_files):
        parser.error("--labels requires one label per data/MC file pair")
    return args


def _axis_title(axis: object) -> str:
    try:
        return str(axis.member("fTitle"))
    except (AttributeError, KeyError):
        return ""


def _theta_axis_is_squared(axis_title: str, histogram_name: str) -> bool:
    """Identify whether the theta axis stores theta squared."""
    normalized = axis_title.lower().replace(" ", "")
    if normalized:
        return any(
            marker in normalized
            for marker in (
                "theta^{2}",
                "theta^2",
                "theta2",
                "squared",
                "deg^{2}",
                "deg^2",
                "deg2",
            )
        )
    return "theta2" in histogram_name.lower()


def _read_histogram(
    root: uproot.ReadOnlyDirectory, name: str, root_file: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    try:
        histogram = root[name]
    except uproot.KeyInFileError as error:
        candidates = [
            str(key).split(";")[0]
            for key in root.keys(recursive=True)
            if "theta" in str(key).lower()
        ]
        raise KeyError(
            f"Histogram {name!r} not found in {root_file}. "
            f"Available theta objects: {candidates}"
        ) from error
    if len(histogram.axes) != 2:
        raise ValueError(f"{name!r} in {root_file} is not two-dimensional")

    energy_edges = np.asarray(histogram.axes[0].edges(), dtype=float)
    theta_edges = np.asarray(histogram.axes[1].edges(), dtype=float)
    theta_squared = _theta_axis_is_squared(_axis_title(histogram.axes[1]), name)
    counts = np.asarray(histogram.values(flow=False), dtype=float)
    histogram_variances = histogram.variances(flow=False)
    if histogram_variances is None:
        raise ValueError(
            f"{name!r} in {root_file} has no stored variances; "
            "ON/OFF uncertainties cannot be inferred from the signed excess"
        )
    variances = np.asarray(histogram_variances, dtype=float)
    expected_shape = (energy_edges.size - 1, theta_edges.size - 1)
    if counts.shape != expected_shape or variances.shape != expected_shape:
        raise ValueError(
            f"Unexpected bin layout for {name!r} in {root_file}: "
            f"counts={counts.shape}, variances={variances.shape}, "
            f"expected={expected_shape}"
        )
    if not np.all(np.isfinite(energy_edges)) or not np.all(np.diff(energy_edges) > 0):
        raise ValueError("The energy axis must have finite, increasing edges")
    if theta_edges[0] < 0 or not np.all(np.diff(theta_edges) > 0):
        raise ValueError("The theta axis must have increasing, non-negative edges")
    if not np.all(np.isfinite(counts)):
        raise ValueError(f"{name!r} in {root_file} contains non-finite counts")
    if np.any(counts < 0):
        warnings.warn(
            f"Using signed ON/OFF-difference contents from {name!r} in {root_file}",
            stacklevel=2,
        )
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError(f"{name!r} in {root_file} contains invalid variances")
    return energy_edges, theta_edges, counts, variances, theta_squared


def read_data(
    root_file: Path, histogram_name: str = DEFAULT_DATA_HISTOGRAM
) -> DataHistogram:
    """Read the signed ON/OFF-difference theta histogram from anasum."""
    with uproot.open(root_file) as root:
        values = _read_histogram(root, histogram_name, root_file)
    return DataHistogram(*values)


def _read_mc_histogram_data(
    root_file: Path, tree_name: str, mc_entry: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with uproot.open(root_file) as root:
        try:
            tree = root[tree_name]
        except uproot.KeyInFileError as error:
            raise KeyError(f"Tree {tree_name!r} not found in {root_file}") from error
        if mc_entry < 0 or mc_entry >= tree.num_entries:
            raise IndexError(
                f"--mc-entry {mc_entry} is outside {tree_name!r} in {root_file} "
                f"(entries: {tree.num_entries})"
            )
        if "IRF" not in tree.keys():
            raise KeyError(f"Tree {tree_name!r} has no IRF branch")
        irf = tree["IRF"].array(
            entry_start=mc_entry, entry_stop=mc_entry + 1, library="np"
        )[0]
        histograms = irf.member("f2DHisto")
        if len(histograms) <= MC_LOGDIFF_HISTOGRAM_INDEX:
            raise ValueError(
                f"IRF object in {root_file} has no hAngularLogDiff_2D histogram"
            )
        histogram = histograms[MC_LOGDIFF_HISTOGRAM_INDEX]
        values, energy_edges, log_theta_edges = histogram.to_numpy(flow=False)
        variances = histogram.variances(flow=False)

    if variances is None:
        raise ValueError(f"hAngularLogDiff_2D in {root_file} has no stored variances")
    return (
        np.asarray(values, dtype=float),
        np.asarray(variances, dtype=float),
        np.asarray(energy_edges, dtype=float),
        np.asarray(log_theta_edges, dtype=float),
    )


def read_mc_histogram(
    root_file: Path,
    tree_name: str = DEFAULT_MC_TREE,
    mc_entry: int = 0,
    max_theta_deg: float | None = DEFAULT_MAX_THETA_DEG,
    max_theta2: float | None = None,
    method: str = "double-gaussian",
    containment: float = DEFAULT_CONTAINMENT,
) -> ContainmentResult:
    """Derive one MC containment radius from the angular histogram."""
    values, variances, energy_edges, log_theta_edges = _read_mc_histogram_data(
        root_file, tree_name, mc_entry
    )
    if values.shape != variances.shape:
        raise ValueError(
            f"MC histogram counts and variances have different shapes in {root_file}"
        )
    if not np.all(np.isfinite(energy_edges)) or not np.all(np.diff(energy_edges) > 0):
        raise ValueError(f"hAngularLogDiff_2D in {root_file} has invalid energy edges")
    if not np.all(np.isfinite(log_theta_edges)) or not np.all(
        np.diff(log_theta_edges) > 0
    ):
        raise ValueError(f"hAngularLogDiff_2D in {root_file} has invalid theta edges")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"hAngularLogDiff_2D in {root_file} contains invalid counts")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError(
            f"hAngularLogDiff_2D in {root_file} contains invalid variances"
        )

    if max_theta2 is not None:
        theta_limit = np.sqrt(max_theta2)
    elif max_theta_deg is None:
        theta_limit = 10.0 ** log_theta_edges[-1]
    else:
        theta_limit = max_theta_deg
    theta_bins = int(
        np.searchsorted(log_theta_edges, np.log10(theta_limit), side="right") - 1
    )
    if theta_bins < 1:
        raise ValueError(
            f"the angular fit limit leaves no MC theta bins in {root_file}"
        )
    theta2_edges = 10.0 ** (2.0 * log_theta_edges[: theta_bins + 1])

    energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    radii = np.full(energy.shape, np.nan)
    uncertainties = np.full((2, energy.size), np.nan)
    for index, (counts, errors) in enumerate(
        zip(values[:, :theta_bins], variances[:, :theta_bins], strict=True)
    ):
        if method == "cumulative":
            radii[index] = _radius(theta2_edges, counts, containment, True)
            continue
        try:
            fit = _fit_psf(theta2_edges, counts, errors, method)
        except ValueError:
            fit = None
        if fit is None:
            continue
        parameters, _ = fit
        radius_function = (
            _double_gaussian_radius if method == "double-gaussian" else _king_radius
        )
        radii[index] = float(radius_function(parameters[None, :], containment)[0])
    return ContainmentResult(energy, radii, uncertainties)


def _radius(
    theta_edges: np.ndarray,
    counts: np.ndarray,
    level: float,
    theta_squared: bool = True,
) -> float:
    # ``counts`` are signed ON/OFF excesses, not Poisson event counts.
    # Keep negative annuli in the cumulative excess profile; clipping them
    # would bias the containment radius outward.
    total = float(np.sum(counts))
    if not np.isfinite(total) or total <= 0:
        return np.nan
    cumulative = np.cumsum(counts)
    crossing = np.flatnonzero(cumulative >= level * total)
    if crossing.size == 0:
        return np.nan
    index = int(crossing[0])
    bin_count = counts[index]
    if bin_count <= 0:
        return np.nan
    previous = cumulative[index - 1] if index else 0.0
    fraction = np.clip((level * total - previous) / bin_count, 0.0, 1.0)
    coordinate = theta_edges[index] + fraction * np.diff(theta_edges)[index]
    return float(np.sqrt(coordinate) if theta_squared else coordinate)


def _rebin_data(data: DataHistogram, factor: int) -> DataHistogram:
    """Combine adjacent energy bins in a data histogram."""
    if factor == 1:
        return data
    starts = np.arange(0, data.counts.shape[0], factor, dtype=int)
    energy_edges = np.concatenate((data.energy_edges[starts], data.energy_edges[-1:]))
    counts = np.add.reduceat(data.counts, starts, axis=0)
    variances = np.add.reduceat(data.variances, starts, axis=0)
    return DataHistogram(
        energy_edges, data.theta_edges, counts, variances, data.theta_squared
    )


def containment_from_data(
    root_file: Path,
    histogram_name: str = DEFAULT_DATA_HISTOGRAM,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 0,
    max_theta_deg: float | None = DEFAULT_MAX_THETA_DEG,
    max_theta2: float | None = None,
    method: str = "double-gaussian",
    energy_rebin: int = 1,
    containment: float = DEFAULT_CONTAINMENT,
) -> ContainmentResult:
    """Calculate one containment radius with a PSF fit or cumulative method."""
    data = _rebin_data(read_data(root_file, histogram_name), energy_rebin)
    energy = 0.5 * (data.energy_edges[:-1] + data.energy_edges[1:])
    radii = np.full(energy.shape, np.nan)
    uncertainties = np.full((2, energy.size), np.nan)

    if max_theta2 is not None:
        coordinate_limit = max_theta2 if data.theta_squared else np.sqrt(max_theta2)
    elif max_theta_deg is None:
        coordinate_limit = data.theta_edges[-1]
    else:
        coordinate_limit = max_theta_deg**2 if data.theta_squared else max_theta_deg
    theta_bins = int(
        np.searchsorted(data.theta_edges, coordinate_limit, side="right") - 1
    )
    if theta_bins < 1:
        raise ValueError(f"the angular fit limit leaves no theta bins in {root_file}")
    theta_edges = data.theta_edges[: theta_bins + 1]
    counts_by_energy = data.counts[:, :theta_bins]
    variances_by_energy = data.variances[:, :theta_bins]

    rng = np.random.default_rng(random_seed)
    if method != "cumulative" and not data.theta_squared:
        raise ValueError("PSF fitting requires a theta-squared histogram")
    for index, (counts, variances) in enumerate(
        zip(counts_by_energy, variances_by_energy, strict=True)
    ):
        if method != "cumulative":
            fit = _fit_psf(theta_edges, counts, variances, method)
            if fit is not None:
                fitted_radii, fitted_uncertainties = _fitted_containment(
                    *fit, method, bootstrap_samples, rng, containment
                )
                radii[index] = fitted_radii
                uncertainties[:, index] = fitted_uncertainties
            continue
        radii[index] = _radius(theta_edges, counts, containment, data.theta_squared)
        central = radii[index]
        if not np.isfinite(central):
            continue
        toys = rng.normal(
            counts, np.sqrt(variances), size=(bootstrap_samples, counts.size)
        )
        toy_radii = np.array(
            [_radius(theta_edges, toy, containment, data.theta_squared) for toy in toys]
        )
        toy_radii = toy_radii[np.isfinite(toy_radii)]
        if toy_radii.size < 100:
            warnings.warn(
                f"Too few valid bootstrap toys for {containment:.0%} containment "
                f"in energy bin {index} of {root_file}",
                stacklevel=2,
            )
            continue
        lower, upper = np.percentile(toy_radii, BOOTSTRAP_PERCENTILES)
        uncertainties[:, index] = (
            max(central - lower, 0.0),
            max(upper - central, 0.0),
        )
    return ContainmentResult(energy, radii, uncertainties)


def default_label(path: Path) -> str:
    return path.parent.name if path.parent.name else path.stem


def _format_radius(value: float, errors: np.ndarray | None = None) -> str:
    if not np.isfinite(value):
        return "--"
    if errors is None or not np.all(np.isfinite(errors)):
        return f"{value:.4f}"
    return f"{value:.4f} -{errors[0]:.4f}/+{errors[1]:.4f}"


def print_containment_tables(
    data: list[ContainmentResult],
    mc: list[ContainmentResult],
    labels: list[str],
    containment: float,
) -> None:
    """Print derived data and MC containment radii as aligned tables."""
    radius_label = f"R{containment:.0%} [deg]"
    for label, data_result, mc_result in zip(labels, data, mc, strict=True):
        print(f"\nContainment radii: {label} (data)")
        print(f"{'log10(Erec/TeV)':>16}  {radius_label:>25}")
        for index, energy in enumerate(data_result.energy_centers):
            print(
                f"{energy:16.4f}  "
                f"{_format_radius(data_result.radii[index], data_result.uncertainties[:, index]):>25}"
            )

        print(f"\nContainment radii: {label} (MC)")
        print(f"{'log10(Erec/TeV)':>16}  {radius_label:>12}")
        for index, energy in enumerate(mc_result.energy_centers):
            print(f"{energy:16.4f}  {_format_radius(mc_result.radii[index]):>12}")


def distribution_from_data(
    root_file: Path,
    histogram_name: str,
    energy_center: float,
    max_theta_deg: float | None = DEFAULT_MAX_THETA_DEG,
    max_theta2: float | None = None,
    energy_rebin: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    data = _rebin_data(read_data(root_file, histogram_name), energy_rebin)
    energy_centers = 0.5 * (data.energy_edges[:-1] + data.energy_edges[1:])
    energy_bin = int(np.argmin(np.abs(energy_centers - energy_center)))
    if max_theta2 is not None:
        coordinate_limit = max_theta2 if data.theta_squared else np.sqrt(max_theta2)
    elif max_theta_deg is None:
        coordinate_limit = data.theta_edges[-1]
    else:
        coordinate_limit = max_theta_deg**2 if data.theta_squared else max_theta_deg
    theta_bins = int(
        np.searchsorted(data.theta_edges, coordinate_limit, side="right") - 1
    )
    if theta_bins < 1:
        raise ValueError(
            f"the angular display limit leaves no theta bins in {root_file}"
        )
    theta_edges_data = data.theta_edges[: theta_bins + 1]
    counts = data.counts[energy_bin, :theta_bins]
    theta_edges = np.sqrt(theta_edges_data) if data.theta_squared else theta_edges_data
    widths = np.diff(theta_edges)
    total = float(np.sum(counts))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"energy bin near {energy_center} in {root_file} has no positive total"
        )
    observed = counts / total / widths
    center = energy_centers[energy_bin]
    display_limit = (
        np.sqrt(coordinate_limit) if data.theta_squared else coordinate_limit
    )
    theta_edges = theta_edges.copy()
    theta_edges[-1] = min(theta_edges[-1], display_limit)
    return theta_edges, observed, center


def distribution_from_mc(
    root_file: Path,
    tree_name: str,
    mc_entry: int,
    energy_center: float,
    max_theta_deg: float = DIAGNOSTIC_MAX_THETA_DEG,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read and normalize one raw MC angular histogram at the nearest energy."""
    values, _, energy_edges, log_theta_edges = _read_mc_histogram_data(
        root_file, tree_name, mc_entry
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"hAngularLogDiff_2D in {root_file} contains invalid counts")
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    energy_bin = int(np.argmin(np.abs(energy_centers - energy_center)))
    theta_bins = int(
        np.searchsorted(log_theta_edges, np.log10(max_theta_deg), side="right") - 1
    )
    if theta_bins < 1:
        raise ValueError(
            f"the angular display limit leaves no MC theta bins in {root_file}"
        )
    theta_edges = 10.0 ** log_theta_edges[: theta_bins + 1]
    counts = values[energy_bin, :theta_bins]
    widths = np.diff(theta_edges)
    total = float(np.sum(counts))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"MC energy bin near {energy_center} in {root_file} has no positive total"
        )
    return theta_edges, counts / total / widths, energy_centers[energy_bin]


def make_plot(
    data: list[ContainmentResult],
    mc: list[ContainmentResult],
    labels: list[str],
    ylim: tuple[float, float] | None,
    energy_range: tuple[float, float] | None = None,
    containment: float = DEFAULT_CONTAINMENT,
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
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    base_colors = ("#0072B2", "#D55E00")
    colors = tuple(
        base_colors[index]
        if index < len(base_colors)
        else plt.get_cmap("tab10")(index % 10)
        for index in range(len(labels))
    )
    data_marker, data_line = "o", "-"
    mc_line = ":"
    for result, label, color in zip(data, labels, colors, strict=True):
        valid = np.isfinite(result.radii)
        axis.errorbar(
            result.energy_centers[valid],
            result.radii[valid],
            yerr=result.uncertainties[:, valid],
            color=color,
            marker=data_marker,
            linestyle=data_line,
            linewidth=1.6,
            markersize=5,
            markerfacecolor="white",
            capsize=1.5,
            elinewidth=0.6,
            label=f"{label} (data, {containment:.0%})",
        )
    for result, label, color in zip(mc, labels, colors, strict=True):
        valid = np.isfinite(result.radii)
        axis.plot(
            result.energy_centers[valid],
            result.radii[valid],
            color=color,
            linestyle=mc_line,
            linewidth=1.5,
            label=f"{label} (MC, {containment:.0%})",
        )
    all_energy = np.concatenate(
        [result.energy_centers for result in (*data, *mc) if result.energy_centers.size]
    )
    energy_min, energy_max = np.min(all_energy), np.max(all_energy)
    margin = 0.03 * (energy_max - energy_min or 1.0)
    axis.set_xlim(
        *(
            energy_range
            if energy_range is not None
            else (energy_min - margin, energy_max + margin)
        )
    )
    axis.set_ylim(*(ylim if ylim is not None else (0, None)))
    axis.set_xlabel(r"$\log_{10}(E_{\mathrm{rec}}/\mathrm{TeV})$")
    axis.set_ylabel("Angular containment radius [deg]")
    axis.grid(color="0.88", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    return figure


def make_distribution_plot(
    distributions: list[
        list[
            tuple[
                tuple[np.ndarray, np.ndarray, float],
                tuple[np.ndarray, np.ndarray, float],
            ]
        ]
    ],
    energy_centers: tuple[float, ...],
    labels: list[str],
    max_theta_deg: float = DIAGNOSTIC_MAX_THETA_DEG,
) -> plt.Figure:
    """Plot data and raw MC histograms in four energy panels."""
    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10.5, 7.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = axes.ravel()
    base_colors = ("#0072B2", "#D55E00")
    colors = tuple(
        base_colors[index]
        if index < len(base_colors)
        else plt.get_cmap("tab10")(index % 10)
        for index in range(len(labels))
    )
    for axis, target, panel in zip(axes, energy_centers, distributions, strict=True):
        for (data_distribution, mc_distribution), label, color in zip(
            panel, labels, colors, strict=True
        ):
            data_edges, observed, _ = data_distribution
            mc_edges, simulation, _ = mc_distribution
            axis.stairs(
                observed,
                data_edges,
                color=color,
                linewidth=1.1,
                alpha=0.8,
                label=f"{label} (data)",
            )
            axis.stairs(
                simulation,
                mc_edges,
                color=color,
                linestyle="--",
                linewidth=1.3,
                label=f"{label} (MC histogram)",
            )
        axis.set_xlim(0, max_theta_deg)
        axis.set_title(
            rf"$\log_{{10}}(E_{{\mathrm{{rec}}}}/\mathrm{{TeV}})\approx {target:.1f}$"
        )
        axis.grid(color="0.88", linewidth=0.6)
    axes[2].set_xlabel(r"Angular separation $\theta$ [deg]")
    axes[3].set_xlabel(r"Angular separation $\theta$ [deg]")
    axes[0].set_ylabel("Normalized density [deg$^{-1}$]")
    axes[2].set_ylabel("Normalized density [deg$^{-1}$]")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=min(3, len(legend_labels)),
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


def main() -> None:
    args = parse_args()
    labels = (
        list(args.labels)
        if args.labels
        else [default_label(path) for path in args.data_files]
    )
    data = [
        containment_from_data(
            path,
            args.data_histogram,
            args.bootstrap_samples,
            args.random_seed + index,
            args.max_theta,
            args.max_theta2,
            args.method,
            args.energy_rebin,
            args.containment,
        )
        for index, path in enumerate(args.data_files)
    ]
    mc_tree = args.mc_tree or _mc_tree_for_containment(args.containment)
    mc = [
        read_mc_histogram(
            path,
            tree_name=mc_tree,
            mc_entry=args.mc_entry,
            max_theta_deg=args.max_theta,
            max_theta2=args.max_theta2,
            method=args.method,
            containment=args.containment,
        )
        for path in args.mc_files
    ]
    print_containment_tables(data, mc, labels, args.containment)
    if args.max_theta is not None:
        diagnostic_max_theta = min(args.max_theta, DIAGNOSTIC_MAX_THETA_DEG)
    elif args.max_theta2 is not None:
        diagnostic_max_theta = min(np.sqrt(args.max_theta2), DIAGNOSTIC_MAX_THETA_DEG)
    else:
        diagnostic_max_theta = DIAGNOSTIC_MAX_THETA_DEG
    distributions = [
        [
            (
                distribution_from_data(
                    data_path,
                    args.data_histogram,
                    target_energy,
                    diagnostic_max_theta,
                    None,
                    args.energy_rebin,
                ),
                distribution_from_mc(
                    mc_path,
                    mc_tree,
                    args.mc_entry,
                    target_energy,
                    DIAGNOSTIC_MAX_THETA_DEG,
                ),
            )
            for data_path, mc_path in zip(args.data_files, args.mc_files, strict=True)
        ]
        for target_energy in DIAGNOSTIC_ENERGY_CENTERS
    ]
    figure = make_plot(
        data,
        mc,
        labels,
        tuple(args.ylim) if args.ylim else None,
        tuple(args.energy_range) if args.energy_range else None,
        args.containment,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")
    distribution_figure = make_distribution_plot(
        distributions,
        DIAGNOSTIC_ENERGY_CENTERS,
        labels,
        DIAGNOSTIC_MAX_THETA_DEG,
    )
    args.distribution_output.parent.mkdir(parents=True, exist_ok=True)
    distribution_figure.savefig(
        args.distribution_output, dpi=args.dpi, bbox_inches="tight"
    )
    plt.close(distribution_figure)
    print(f"Wrote {args.distribution_output}")


def _double_gaussian_bin_counts(
    bounds, norm, fraction, sigma_core, sigma_offset, background
):
    lower, upper = bounds
    sigma_tail = sigma_core + sigma_offset
    return norm * (
        fraction
        * (np.exp(-lower / (2 * sigma_core**2)) - np.exp(-upper / (2 * sigma_core**2)))
        + (1 - fraction)
        * (np.exp(-lower / (2 * sigma_tail**2)) - np.exp(-upper / (2 * sigma_tail**2)))
    ) + background * (upper - lower)


def _king_bin_counts(bounds, norm, sigma, gamma, background):
    lower, upper = bounds
    exponent = 1 - gamma
    source = (1 + lower / (2 * gamma * sigma**2)) ** exponent - (
        1 + upper / (2 * gamma * sigma**2)
    ) ** exponent
    return norm * source + background * (upper - lower)


def _double_gaussian_radius(parameters: np.ndarray, level: float) -> np.ndarray:
    fraction, sigma_core, sigma_offset = (
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3],
    )
    sigma_tail = sigma_core + sigma_offset
    target = 1 - level
    lower = np.zeros(parameters.shape[0])
    upper = 200 * sigma_tail**2
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        tail = fraction * np.exp(-middle / (2 * sigma_core**2)) + (
            1 - fraction
        ) * np.exp(-middle / (2 * sigma_tail**2))
        lower = np.where(tail > target, middle, lower)
        upper = np.where(tail > target, upper, middle)
    return np.sqrt(0.5 * (lower + upper))


def _king_radius(parameters: np.ndarray, level: float) -> np.ndarray:
    sigma, gamma = parameters[:, 1], parameters[:, 2]
    theta2 = 2 * gamma * sigma**2 * ((1 - level) ** (1 / (1 - gamma)) - 1)
    return np.sqrt(theta2)


def _fit_psf(
    theta_edges: np.ndarray, counts: np.ndarray, variances: np.ndarray, method: str
):
    if method not in {"double-gaussian", "king"}:
        raise ValueError(f"Unsupported PSF fit method: {method}")
    lower, upper = theta_edges[:-1], theta_edges[1:]
    usable = np.isfinite(counts) & np.isfinite(variances) & (variances > 0)
    if np.count_nonzero(usable) < (5 if method == "double-gaussian" else 4):
        raise ValueError("Too few bins with positive finite variances for PSF fit")
    bounds = (lower[usable], upper[usable])
    density = counts[usable] / (upper[usable] - lower[usable])
    background = float(np.median(density[-max(1, density.size // 4) :]))
    norm = max(
        float(np.sum(counts[usable] - background * (upper[usable] - lower[usable]))),
        1e-6,
    )
    if method == "double-gaussian":
        function = _double_gaussian_bin_counts
        initial = (norm, 0.7, 0.03, 0.07, background)
        parameter_bounds = ((0, 0, 1e-4, 0, -np.inf), (np.inf, 1, 1, 1, np.inf))
    else:
        function = _king_bin_counts
        initial = (norm, 0.05, 2.5, background)
        parameter_bounds = ((0, 1e-4, 1.001, -np.inf), (np.inf, 1, 100, np.inf))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            parameters, covariance = curve_fit(
                function,
                bounds,
                counts[usable],
                p0=initial,
                sigma=np.sqrt(variances[usable]),
                absolute_sigma=True,
                bounds=parameter_bounds,
                maxfev=50000,
            )
    except (RuntimeError, ValueError, OptimizeWarning) as error:
        warnings.warn(f"{method} PSF fit failed: {error}", stacklevel=2)
        return None
    if not np.all(np.isfinite(covariance)):
        warnings.warn(
            f"{method} PSF fit has no finite covariance estimate", stacklevel=2
        )
        covariance = None
    return parameters, covariance


def _fitted_containment(
    parameters: np.ndarray,
    covariance: np.ndarray | None,
    method: str,
    samples: int,
    rng: np.random.Generator,
    containment: float,
):
    radius_function = (
        _double_gaussian_radius if method == "double-gaussian" else _king_radius
    )
    central_parameters = parameters[None, :]
    radius = float(radius_function(central_parameters, containment)[0])
    uncertainty = np.array([np.nan, np.nan])
    if covariance is None:
        return radius, uncertainty
    toys = rng.multivariate_normal(parameters, covariance, size=samples)
    if method == "double-gaussian":
        valid = (
            (toys[:, 0] >= 0)
            & (toys[:, 1] >= 0)
            & (toys[:, 1] <= 1)
            & (toys[:, 2] > 0)
            & (toys[:, 3] >= 0)
        )
    else:
        valid = (toys[:, 0] >= 0) & (toys[:, 1] > 0) & (toys[:, 2] > 1)
    toy_radii = radius_function(toys[valid], containment)
    if toy_radii.size >= 100:
        lower, upper = np.percentile(toy_radii, BOOTSTRAP_PERCENTILES)
        uncertainty = (
            max(radius - lower, 0),
            max(upper - radius, 0),
        )
    return radius, uncertainty


if __name__ == "__main__":
    main()
