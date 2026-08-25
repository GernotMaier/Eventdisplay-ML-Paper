# Eventdisplay-ML documents

This repository contains documents related to the [Eventdisplay-ML](https://github.com/Eventdisplay/Eventdisplay-ML) project.

## Overview

Eventdisplay-ML is a machine learning extension for the Eventdisplay framework, which is used for the analysis of very-high-energy gamma-ray data.

The repository contains two LaTeX documents:

- `stereo-reconstruction/`: describing the stereo reconstruction of gamma-ray events
- `classification/`: describing the classification of gamma-ray events

## Building

Build the stereo-reconstruction paper:

```sh
cd stereo-reconstruction
make
```

Build the classification:

```sh
cd classification
make
```

## Python figures

Create and activate the conda environment used by the plotting scripts:

```sh
conda env create -f environment.yml
conda activate eventdisplay-ml-paper
```

To bring an existing environment back in sync with the file, run
`conda env update -f environment.yml --prune`.

### Angular resolution and containment radii

`plot_angular_containment_anasum.py` compares background-subtracted anasum
data with MC containment derived from the IRF's
`hAngularLogDiff_2D` histogram. It does not use the precomputed angular-
resolution branches in `fEffArea`. Give one anasum file and one matching IRF
file for each reconstruction:

```sh
cd scripts
python ./plot_angular_containment_anasum.py \
    ../tmp-data/anasum_tmva.root ../tmp-data/anasum_xgb.root \
    --mc-files ../tmp-data/eff_area_tmva.root ../tmp-data/eff_area_xgb.root \
    --labels TMVA XGB \
    --ylim 0. 0.4 \
    --energy-bin 10 \
    --max-theta 0.5 \
    --energy-rebin 2 \
    --energy-range -1. 1.5
```

By default, the data histogram is
`total_1/stereo/stereoParameterHistograms/htheta2Erec_diff`; it must be a TH2
with reconstructed log-energy on x, theta-squared on y, and stored bin
variances. The script fits each energy bin up to 0.5 degrees with a
double-Gaussian PSF and reports one configurable containment radius for the
fitted source component. The default is 68%; use, for example,
`--containment 0.95` (or `--containment 95`) for 95%. The MC tree is selected
from this value by default: 68% uses `t_angular_resolution`, while other
integer percentages use the corresponding `_NNNp` suffix, such as
`t_angular_resolution_095p`. Use `--mc-tree` to override that selection. Use
`--method king` for a King-profile fit or `--method cumulative` for
non-parametric containment of the signed excess. `--max-theta` and
`--max-theta2` set the fit range; `--energy-rebin` combines adjacent data
energy bins only.

For a fit diagnostic, add `--energy-bin INDEX` to show the data distribution
and fitted PSF for that zero-based (and, if rebinned, rebinned) data bin.
Uncertainties for fitted data radii are 68% intervals propagated from the fit
covariance using reproducible parameter toys. The MC curves are fitted from
their histogram but currently have no uncertainty bands. `--mc-entry` selects
the IRF-tree entry (normally an azimuth bin). Use `--mc-tree` or
`--data-histogram` only when the ROOT object paths differ from the defaults.

## License

This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [LICENSE](LICENSE) file for details.

## Related Projects

- [Eventdisplay-ML](https://github.com/Eventdisplay/Eventdisplay-ML) - Machine Learning for Eventdisplay
