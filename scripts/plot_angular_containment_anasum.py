#!/usr/bin/env python3
"""Compare anasum angular containment with IRF angular resolution.

Data are read from the background-subtracted ``htheta2Erec_diff`` histogram
written by anasum.  The Monte Carlo curves are read directly from the
``Rec_angRes_p68`` and ``Rec_angRes_p80`` branches of the ``fEffArea`` tree.
The energy axis of the histogram is expected to be log10(E/TeV), as in the
other angular-containment plotting scripts in this repository.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot

DEFAULT_DATA_HISTOGRAM = (
    "total_1/stereo/stereoParameterHistograms/htheta2Erec_diff"
)
DEFAULT_MC_TREE = "fEffArea"
DEFAULT_MC_P68_BRANCH = "Rec_angRes_p68"
DEFAULT_MC_P80_BRANCH = "Rec_angRes_p80"
CONTAINMENT_LEVELS = (0.68, 0.80)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 68% and 80% angular containment from two anasum files "
            "with the corresponding fEffArea IRF curves."
        )
    )
    parser.add_argument(
        "data_files",
        nargs=2,
        type=Path,
        metavar="ANASUM_FILE",
        help="the two anasum output files to compare",
    )
    parser.add_argument(
        "--mc-files",
        nargs=2,
        type=Path,
        required=True,
        metavar=("IRF_FILE_1", "IRF_FILE_2"),
        help="the matching instrument-response files containing fEffArea",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        metavar=("LABEL_1", "LABEL_2"),
        help="legend labels (default: parent directory names)",
    )
    parser.add_argument(
        "--data-histogram",
        default=DEFAULT_DATA_HISTOGRAM,
        help=f"anasum histogram (default: {DEFAULT_DATA_HISTOGRAM})",
    )
    parser.add_argument(
        "--mc-tree",
        default=DEFAULT_MC_TREE,
        help=f"IRF tree (default: {DEFAULT_MC_TREE})",
    )
    parser.add_argument(
        "--mc-energy-branch",
        help="energy branch in fEffArea (default: automatic detection)",
    )
    parser.add_argument(
        "--mc-entry",
        type=int,
        default=0,
        help="fEffArea tree entry to use when no energy branch is present (default: %(default)s)",
    )
    energy_group = parser.add_mutually_exclusive_group()
    energy_group.add_argument(
        "--mc-energy-log10",
        action="store_true",
        help="interpret the MC energy branch as log10(E/TeV)",
    )
    energy_group.add_argument(
        "--mc-energy-linear",
        action="store_true",
        help="interpret the MC energy branch as E in TeV",
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
        "-o",
        "--output",
        type=Path,
        default=Path("angular_containment_anasum_comparison.pdf"),
        help="output plot (default: %(default)s)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.ylim is not None and args.ylim[0] >= args.ylim[1]:
        parser.error("--ylim requires YMIN < YMAX")
    return args


def _read_histogram(
    root: uproot.ReadOnlyDirectory, name: str, root_file: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    counts = np.asarray(histogram.values(flow=False), dtype=float)
    histogram_variances = histogram.variances(flow=False)
    if histogram_variances is None:
        warnings.warn(
            f"{name!r} in {root_file} has no stored variances; "
            "using abs(counts) for the data bootstrap",
            stacklevel=2,
        )
        variances = np.abs(counts)
    else:
        variances = np.asarray(histogram_variances, dtype=float)
    expected_shape = (energy_edges.size - 1, theta_edges.size - 1)
    if counts.shape != expected_shape or variances.shape != expected_shape:
        raise ValueError(
            f"Unexpected bin layout for {name!r} in {root_file}: "
            f"counts={counts.shape}, variances={variances.shape}, "
            f"expected={expected_shape}"
        )
    if theta_edges[0] < 0 or not np.all(np.diff(theta_edges) > 0):
        raise ValueError("The theta axis must have increasing, non-negative edges")
    if not np.all(np.isfinite(counts)):
        raise ValueError(f"{name!r} in {root_file} contains non-finite counts")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError(f"{name!r} in {root_file} contains invalid variances")
    return energy_edges, theta_edges, counts, variances


def read_data(
    root_file: Path, histogram_name: str = DEFAULT_DATA_HISTOGRAM
) -> DataHistogram:
    """Read the background-subtracted theta histogram from an anasum file."""
    with uproot.open(root_file) as root:
        values = _read_histogram(root, histogram_name, root_file)
    return DataHistogram(*values)


def _branch_name(
    available: list[str], requested: str | None, candidates: tuple[str, ...], label: str
) -> str:
    if requested is not None:
        if requested not in available:
            raise KeyError(f"Branch {requested!r} not found; available branches: {available}")
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


def read_mc(
    root_file: Path,
    tree_name: str = DEFAULT_MC_TREE,
    energy_branch: str | None = None,
    energy_is_log10: bool | None = None,
    mc_entry: int = 0,
) -> ContainmentResult:
    """Read one fEffArea IRF row and its p68/p80 angular resolutions.

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
        energy_candidates = ("Erec", "E_reco", "Energy", "energy", "E", "logE", "log10E")
        if energy_branch is not None:
            energy_name = _branch_name(
                available, energy_branch, energy_candidates, "energy"
            )
        else:
            energy_name = next(
                (by_lower[candidate.lower()] for candidate in energy_candidates
                 if candidate.lower() in by_lower),
                None,
            )
        p68_name = _branch_name(
            available, DEFAULT_MC_P68_BRANCH, (DEFAULT_MC_P68_BRANCH,), "68% resolution"
        )
        p80_name = _branch_name(
            available, DEFAULT_MC_P80_BRANCH, (DEFAULT_MC_P80_BRANCH,), "80% resolution"
        )
        branch_names = [p68_name, p80_name]
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
            p80 = np.asarray(arrays[p80_name][0], dtype=float)
            energy_is_log10 = True
        else:
            energy = np.asarray(arrays[energy_name][0], dtype=float)
            p68 = np.asarray(arrays[p68_name][0], dtype=float)
            p80 = np.asarray(arrays[p80_name][0], dtype=float)
            title = _branch_title(tree, energy_name)

    if not (energy.shape == p68.shape == p80.shape) or energy.ndim != 1:
        raise ValueError(
            f"MC branches in entry {mc_entry} of {root_file} do not contain "
            "matching 1D arrays"
        )
    if energy_is_log10 is None:
        energy_is_log10 = (
            "log" in f"{energy_name} {title}".lower() or np.any(energy <= 0)
        )
    if not energy_is_log10:
        if np.any(energy <= 0):
            raise ValueError(
                f"MC energy branch {energy_name!r} contains non-positive values"
            )
        energy = np.log10(energy)
    if not np.all(np.isfinite(energy)):
        raise ValueError("MC energy values contain non-finite values")
    for name, values in ((p68_name, p68), (p80_name, p80)):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"MC branch {name!r} contains invalid angular resolutions")

    order = np.argsort(energy)
    energy = energy[order]
    radii = {0.68: p68[order], 0.80: p80[order]}
    uncertainties = {
        level: np.full((2, energy.size), np.nan) for level in CONTAINMENT_LEVELS
    }
    return ContainmentResult(energy, radii, uncertainties)


def _radius(theta_edges: np.ndarray, counts: np.ndarray, level: float) -> float:
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
    return float(np.sqrt(theta_edges[index] + fraction * np.diff(theta_edges)[index]))


def containment_from_data(
    root_file: Path,
    histogram_name: str = DEFAULT_DATA_HISTOGRAM,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    random_seed: int = 0,
) -> ContainmentResult:
    """Calculate data containment and Gaussian-toy confidence intervals."""
    data = read_data(root_file, histogram_name)
    energy = 0.5 * (data.energy_edges[:-1] + data.energy_edges[1:])
    radii = {level: np.full(energy.shape, np.nan) for level in CONTAINMENT_LEVELS}
    uncertainties = {
        level: np.full((2, energy.size), np.nan) for level in CONTAINMENT_LEVELS
    }
    rng = np.random.default_rng(random_seed)
    for index, (counts, variances) in enumerate(zip(data.counts, data.variances, strict=True)):
        for level in CONTAINMENT_LEVELS:
            radii[level][index] = _radius(data.theta_edges, counts, level)
            central = radii[level][index]
            if not np.isfinite(central):
                continue
            toys = rng.normal(
                counts, np.sqrt(variances), size=(bootstrap_samples, counts.size)
            )
            toy_radii = np.array(
                [_radius(data.theta_edges, toy, level) for toy in toys]
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


def make_plot(
    data: list[ContainmentResult],
    mc: list[ContainmentResult],
    labels: list[str],
    ylim: tuple[float, float] | None,
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
    colors = ("#0072B2", "#D55E00")
    data_styles = {0.68: ("o", "-"), 0.80: ("s", "--")}
    mc_styles = {0.68: (":",), 0.80: ("-.",)}
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
    axis.set_xlim(energy_min - margin, energy_max + margin)
    axis.set_ylim(*(ylim if ylim is not None else (0, None)))
    axis.set_xlabel(r"$\log_{10}(E_{\mathrm{rec}}/\mathrm{TeV})$")
    axis.set_ylabel("Angular containment radius [deg]")
    axis.grid(color="0.88", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    return figure


def main() -> None:
    args = parse_args()
    labels = list(args.labels) if args.labels else [default_label(path) for path in args.data_files]
    energy_is_log10 = True if args.mc_energy_log10 else False if args.mc_energy_linear else None
    data = [
        containment_from_data(
            path,
            args.data_histogram,
            args.bootstrap_samples,
            args.random_seed + index,
        )
        for index, path in enumerate(args.data_files)
    ]
    mc = [
        read_mc(
            path,
            args.mc_tree,
            args.mc_energy_branch,
            energy_is_log10,
            args.mc_entry,
        )
        for path in args.mc_files
    ]
    figure = make_plot(data, mc, labels, tuple(args.ylim) if args.ylim else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
