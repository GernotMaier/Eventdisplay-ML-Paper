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
DEFAULT_MC_P68_BRANCH = "Rec_angRes_p68"
DEFAULT_MC_P95_BRANCH = "Rec_angRes_p80"
DEFAULT_MAX_THETA_DEG = 0.5
CONTAINMENT_LEVELS = (0.68, 0.95)
DEFAULT_BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_PERCENTILES = (15.865, 84.135)


@dataclass(frozen=True)
class ContainmentResult:
    energy_centers: np.ndarray
    radii: dict[float, np.ndarray]
    uncertainties: dict[float, np.ndarray]


@dataclass(frozen=True)
class DataHistogram:
    energy_edges: np.ndarray
    theta_edges: np.ndarray
    counts: np.ndarray
    variances: np.ndarray
    theta_squared: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 68% and 95% angular containment from one or more anasum files "
            "with curves derived from the corresponding IRF objects."
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
        "--mc-tree",
        default=DEFAULT_MC_TREE,
        help=f"IRF tree containing hAngularLogDiff_2D (default: {DEFAULT_MC_TREE})",
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
        "--energy-bin",
        type=int,
        metavar="INDEX",
        help="show the 1D theta distribution and 2D PSF fit for this zero-based energy-bin index",
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
    if args.energy_bin is not None and args.energy_bin < 0:
        parser.error("--energy-bin must be non-negative")
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


def _branch_name(
    available: list[str], requested: str | None, candidates: tuple[str, ...], label: str
) -> str:
    if requested is not None:
        if requested not in available:
            raise KeyError(
                f"Branch {requested!r} not found; available branches: {available}"
            )
        return requested
    by_lower = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise KeyError(f"Could not find {label} branch; available branches: {available}")


def _branch_title(tree: uproot.behaviors.TBranch.TBranch, name: str) -> str:
    try:
        return str(tree[name].title)
    except (AttributeError, KeyError):
        return ""


def read_mc_histogram(
    root_file: Path,
    tree_name: str = DEFAULT_MC_TREE,
    mc_entry: int = 0,
    max_theta_deg: float | None = DEFAULT_MAX_THETA_DEG,
    max_theta2: float | None = None,
    method: str = "double-gaussian",
) -> ContainmentResult:
    """Derive MC containment from the reconstructed-energy angular histogram."""
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
    values = np.asarray(values, dtype=float)
    variances = np.asarray(variances, dtype=float)
    energy_edges = np.asarray(energy_edges, dtype=float)
    log_theta_edges = np.asarray(log_theta_edges, dtype=float)
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
    radii = {level: np.full(energy.shape, np.nan) for level in CONTAINMENT_LEVELS}
    uncertainties = {
        level: np.full((2, energy.size), np.nan) for level in CONTAINMENT_LEVELS
    }
    for index, (counts, errors) in enumerate(
        zip(values[:, :theta_bins], variances[:, :theta_bins], strict=True)
    ):
        if method == "cumulative":
            for level in CONTAINMENT_LEVELS:
                radii[level][index] = _radius(theta2_edges, counts, level, True)
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
        for level in CONTAINMENT_LEVELS:
            radii[level][index] = float(radius_function(parameters[None, :], level)[0])
    return ContainmentResult(energy, radii, uncertainties)


def read_mc(
    root_file: Path,
    tree_name: str = DEFAULT_MC_TREE,
    energy_branch: str | None = None,
    energy_is_log10: bool | None = None,
    mc_entry: int = 0,
) -> ContainmentResult:
    """Read one fEffArea IRF row and its p68/p95 angular resolutions.

    fEffArea files produced by Eventdisplay do not have a standalone energy
    branch.  Their reconstructed-energy centers are stored in the variable-
    length Rec_e0 branch, with Rec_nbins describing its length.
    """
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
        available = [str(name) for name in tree.keys()]
        by_lower = {name.lower(): name for name in available}
        energy_candidates = (
            "Erec",
            "E_reco",
            "Energy",
            "energy",
            "E",
            "logE",
            "log10E",
        )
        if energy_branch is not None:
            energy_name = _branch_name(
                available, energy_branch, energy_candidates, "energy"
            )
        else:
            energy_name = next(
                (
                    by_lower[candidate.lower()]
                    for candidate in energy_candidates
                    if candidate.lower() in by_lower
                ),
                None,
            )
        p68_name = _branch_name(
            available, DEFAULT_MC_P68_BRANCH, (DEFAULT_MC_P68_BRANCH,), "68% resolution"
        )
        p95_name = _branch_name(
            available, DEFAULT_MC_P95_BRANCH, (DEFAULT_MC_P95_BRANCH,), "95% resolution"
        )
        branch_names = [p68_name, p95_name]
        if energy_name is None:
            for metadata_name in ("Rec_e0", "Rec_nbins"):
                if metadata_name not in available:
                    raise KeyError(
                        f"fEffArea has no energy branch and no {metadata_name!r} metadata; "
                        f"available branches: {available}"
                    )
            branch_names = ["Rec_e0", "Rec_nbins", *branch_names]
        else:
            branch_names = [energy_name, *branch_names]
        arrays = tree.arrays(
            branch_names,
            entry_start=mc_entry,
            entry_stop=mc_entry + 1,
            library="np",
        )
        if energy_name is None:
            energy = np.asarray(arrays["Rec_e0"][0], dtype=float)
            p68 = np.asarray(arrays[p68_name][0], dtype=float)
            p95 = np.asarray(arrays[p95_name][0], dtype=float)
            energy_is_log10 = True
        else:
            energy = np.asarray(arrays[energy_name][0], dtype=float)
            p68 = np.asarray(arrays[p68_name][0], dtype=float)
            p95 = np.asarray(arrays[p95_name][0], dtype=float)
            title = _branch_title(tree, energy_name)

    if not (energy.shape == p68.shape == p95.shape) or energy.ndim != 1:
        raise ValueError(
            f"MC branches in entry {mc_entry} of {root_file} do not contain "
            "matching 1D arrays"
        )
    if energy_is_log10 is None:
        energy_is_log10 = "log" in f"{energy_name} {title}".lower() or np.any(
            energy <= 0
        )
    if not energy_is_log10:
        if np.any(energy <= 0):
            raise ValueError(
                f"MC energy branch {energy_name!r} contains non-positive values"
            )
        energy = np.log10(energy)
    if not np.all(np.isfinite(energy)):
        raise ValueError("MC energy values contain non-finite values")
    for name, values in ((p68_name, p68), (p95_name, p95)):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"MC branch {name!r} contains invalid angular resolutions")

    order = np.argsort(energy)
    energy = energy[order]
    radii = {0.68: p68[order], 0.95: p95[order]}
    uncertainties = {
        level: np.full((2, energy.size), np.nan) for level in CONTAINMENT_LEVELS
    }
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
) -> ContainmentResult:
    """Calculate containment with a PSF fit or the legacy cumulative method."""
    data = _rebin_data(read_data(root_file, histogram_name), energy_rebin)
    energy = 0.5 * (data.energy_edges[:-1] + data.energy_edges[1:])
    radii = {level: np.full(energy.shape, np.nan) for level in CONTAINMENT_LEVELS}
    uncertainties = {
        level: np.full((2, energy.size), np.nan) for level in CONTAINMENT_LEVELS
    }

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
                    *fit, method, bootstrap_samples, rng
                )
                for level in CONTAINMENT_LEVELS:
                    radii[level][index] = fitted_radii[level]
                    uncertainties[level][:, index] = fitted_uncertainties[level]
            continue
        for level in CONTAINMENT_LEVELS:
            radii[level][index] = _radius(
                theta_edges, counts, level, data.theta_squared
            )
            central = radii[level][index]
            if not np.isfinite(central):
                continue
            toys = rng.normal(
                counts, np.sqrt(variances), size=(bootstrap_samples, counts.size)
            )
            toy_radii = np.array(
                [_radius(theta_edges, toy, level, data.theta_squared) for toy in toys]
            )
            toy_radii = toy_radii[np.isfinite(toy_radii)]
            if toy_radii.size < 100:
                warnings.warn(
                    f"Too few valid bootstrap toys for {level:.0%} containment "
                    f"in energy bin {index} of {root_file}",
                    stacklevel=2,
                )
                continue
            lower, upper = np.percentile(toy_radii, BOOTSTRAP_PERCENTILES)
            uncertainties[level][:, index] = (
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
) -> None:
    """Print derived data and MC containment radii as aligned tables."""
    for label, data_result, mc_result in zip(labels, data, mc, strict=True):
        print(f"\nContainment radii: {label} (data)")
        print(f"{'log10(Erec/TeV)':>16}  {'R68 [deg]':>25}  {'R95 [deg]':>25}")
        for index, energy in enumerate(data_result.energy_centers):
            print(
                f"{energy:16.4f}  "
                f"{_format_radius(data_result.radii[0.68][index], data_result.uncertainties[0.68][:, index]):>25}  "
                f"{_format_radius(data_result.radii[0.95][index], data_result.uncertainties[0.95][:, index]):>25}"
            )

        print(f"\nContainment radii: {label} (MC)")
        print(f"{'log10(Erec/TeV)':>16}  {'R68 [deg]':>12}  {'R95 [deg]':>12}")
        for index, energy in enumerate(mc_result.energy_centers):
            print(
                f"{energy:16.4f}  "
                f"{_format_radius(mc_result.radii[0.68][index]):>12}  "
                f"{_format_radius(mc_result.radii[0.95][index]):>12}"
            )


def distribution_from_data(
    root_file: Path,
    histogram_name: str,
    energy_bin: int,
    method: str,
    max_theta_deg: float | None = DEFAULT_MAX_THETA_DEG,
    max_theta2: float | None = None,
    energy_rebin: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    data = _rebin_data(read_data(root_file, histogram_name), energy_rebin)
    if energy_bin >= data.counts.shape[0]:
        raise ValueError(
            f"--energy-bin {energy_bin} is outside {root_file} (bins: {data.counts.shape[0]})"
        )
    if method == "cumulative":
        raise ValueError(
            "--energy-bin requires a fitted method (double-gaussian or king)"
        )
    if not data.theta_squared:
        raise ValueError("PSF fitting requires a theta-squared histogram")
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
    theta_edges_data = data.theta_edges[: theta_bins + 1]
    counts = data.counts[energy_bin, :theta_bins]
    variances = data.variances[energy_bin, :theta_bins]
    fit = _fit_psf(theta_edges_data, counts, variances, method)
    if fit is None:
        raise ValueError(
            f"2D PSF fit failed for energy bin {energy_bin} in {root_file}"
        )
    parameters, _ = fit
    theta_edges = np.sqrt(theta_edges_data) if data.theta_squared else theta_edges_data
    widths = np.diff(theta_edges)
    total = float(np.sum(counts))
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"energy bin {energy_bin} in {root_file} has no positive total"
        )
    observed = counts / total / widths
    bounds = (theta_edges_data[:-1], theta_edges_data[1:])
    model_counts = (
        _double_gaussian_bin_counts(bounds, *parameters)
        if method == "double-gaussian"
        else _king_bin_counts(bounds, *parameters)
    )
    model_total = float(np.sum(model_counts))
    if not np.isfinite(model_total) or model_total <= 0:
        raise ValueError(
            f"2D PSF fit has no positive model total in energy bin {energy_bin} of {root_file}"
        )
    fitted = model_counts / model_total / widths
    center = 0.5 * (data.energy_edges[energy_bin] + data.energy_edges[energy_bin + 1])
    display_limit = (
        np.sqrt(coordinate_limit) if data.theta_squared else coordinate_limit
    )
    theta_edges = theta_edges.copy()
    theta_edges[-1] = min(theta_edges[-1], display_limit)
    return (
        theta_edges,
        observed,
        fitted,
        rf"$\log_{{10}}(E_{{\mathrm{{rec}}}}/\mathrm{{TeV}}) = {center:.3f}$",
    )


def make_plot(
    data: list[ContainmentResult],
    mc: list[ContainmentResult],
    labels: list[str],
    ylim: tuple[float, float] | None,
    energy_range: tuple[float, float] | None = None,
    distributions: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] | None = None,
    method: str = "double-gaussian",
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
    if distributions is None:
        figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
        distribution_axis = None
    else:
        figure, (axis, distribution_axis) = plt.subplots(
            ncols=2,
            figsize=(10.5, 4.5),
            gridspec_kw={"width_ratios": (1.6, 1.0)},
            constrained_layout=True,
        )
    base_colors = ("#0072B2", "#D55E00")
    colors = tuple(
        base_colors[index]
        if index < len(base_colors)
        else plt.get_cmap("tab10")(index % 10)
        for index in range(len(labels))
    )
    data_styles = {0.68: ("o", "-"), 0.95: ("s", "--")}
    mc_styles = {0.68: (":",), 0.95: ("-.",)}
    for result, label, color in zip(data, labels, colors, strict=True):
        for level in CONTAINMENT_LEVELS:
            marker, line = data_styles[level]
            valid = np.isfinite(result.radii[level])
            axis.errorbar(
                result.energy_centers[valid],
                result.radii[level][valid],
                yerr=result.uncertainties[level][:, valid],
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.6,
                markersize=5,
                markerfacecolor="white",
                capsize=1.5,
                elinewidth=0.6,
                label=f"{label} (data, {level:.0%})",
            )
    for result, label, color in zip(mc, labels, colors, strict=True):
        for level in CONTAINMENT_LEVELS:
            (line,) = mc_styles[level]
            valid = np.isfinite(result.radii[level])
            axis.plot(
                result.energy_centers[valid],
                result.radii[level][valid],
                color=color,
                linestyle=line,
                linewidth=1.5,
                label=f"{label} (MC, {level:.0%})",
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
    if distribution_axis is not None:
        for (theta_edges, observed, fitted, title), label, color in zip(
            distributions, labels, colors, strict=True
        ):
            distribution_axis.stairs(
                observed,
                theta_edges,
                color=color,
                linewidth=1.2,
                label=f"{label} (data)",
            )
            distribution_axis.stairs(
                fitted,
                theta_edges,
                color=color,
                linewidth=1.7,
                label=f"{label} ({method} fit)",
            )
            distribution_axis.set_xlim(0, theta_edges[-1])
            distribution_axis.set_xlabel(r"Angular separation $\theta$ [deg]")
            distribution_axis.set_ylabel("Normalized density [deg$^{-1}$]")
            distribution_axis.set_title(title)
            distribution_axis.grid(color="0.88", linewidth=0.6)
            distribution_axis.legend(frameon=False, fontsize=7)
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
        )
        for index, path in enumerate(args.data_files)
    ]
    mc = [
        read_mc_histogram(
            path,
            args.mc_tree,
            args.mc_entry,
            args.max_theta,
            args.max_theta2,
            args.method,
        )
        for path in args.mc_files
    ]
    print_containment_tables(data, mc, labels)
    distributions = None
    if args.energy_bin is not None:
        distributions = [
            distribution_from_data(
                path,
                args.data_histogram,
                args.energy_bin,
                args.method,
                args.max_theta,
                args.max_theta2,
                args.energy_rebin,
            )
            for path in args.data_files
        ]
    figure = make_plot(
        data,
        mc,
        labels,
        tuple(args.ylim) if args.ylim else None,
        tuple(args.energy_range) if args.energy_range else None,
        distributions,
        args.method,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


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
):
    radius_function = (
        _double_gaussian_radius if method == "double-gaussian" else _king_radius
    )
    radii, uncertainties = {}, {}
    central_parameters = parameters[None, :]
    for level in CONTAINMENT_LEVELS:
        radii[level] = float(radius_function(central_parameters, level)[0])
        uncertainties[level] = np.array([np.nan, np.nan])
        if covariance is None:
            continue
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
        toy_radii = radius_function(toys[valid], level)
        if toy_radii.size >= 100:
            lower, upper = np.percentile(toy_radii, BOOTSTRAP_PERCENTILES)
            uncertainties[level] = (
                max(radii[level] - lower, 0),
                max(upper - radii[level], 0),
            )
    return radii, uncertainties


if __name__ == "__main__":
    main()
