# Eventdisplay-ML documents

This repository contains documents related to the [Eventdisplay-ML](https://github.com/Eventdisplay/Eventdisplay-ML) project.

## Overview

Eventdisplay-ML is a machine learning extension for the Eventdisplay framework, which is used for the analysis of very-high-energy gamma-ray data.

The repository contains two LaTeX documents:

- `stereo-reconstruction-paper/`: the main paper. This paper is in the writing phase and is scoped to stereo reconstruction of direction and energy.
- `classification-note/`: a separate note holding the gamma/hadron separation material that was moved out of the main paper.

## Building

Build the stereo-reconstruction paper:

```sh
cd stereo-reconstruction-paper
make
```

Build the classification note:

```sh
cd classification-note
make
```

Generated LaTeX build products, including PDFs, are ignored by Git.

## License

This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [LICENSE](LICENSE) file for details.

## Related Projects

- [Eventdisplay-ML](https://github.com/Eventdisplay/Eventdisplay-ML) - Machine Learning for Eventdisplay
