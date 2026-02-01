# Eventdisplay-ML-Paper

This repository contains the paper related to the [Eventdisplay-ML](https://github.com/Eventdisplay/Eventdisplay-ML) project.

## Overview

Eventdisplay-ML is a machine learning extension for the Eventdisplay framework, which is used for the analysis of very-high-energy gamma-ray data. This paper documents the methodologies, results, and findings from the machine learning approaches applied to Eventdisplay.

## Building the Paper

This repository includes a LaTeX template for the paper. To compile the paper, you need a LaTeX distribution installed on your system.

### Prerequisites

- LaTeX distribution (e.g., TeX Live, MiKTeX, or MacTeX)
- Make utility (optional, but recommended)

### Compilation

Use the provided Makefile to compile the paper:

```bash
make          # Compile the PDF
make clean    # Remove auxiliary files
make distclean # Remove all generated files including PDF
make view     # Open the compiled PDF (on systems with a PDF viewer)
make help     # Show available commands
```

Alternatively, you can compile manually:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Files

- `main.tex` - Main LaTeX document
- `bibliography.bib` - BibTeX bibliography file
- `Makefile` - Build automation for LaTeX compilation
- `LICENSE` - CC-BY-4.0 license

## License

This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0). See the [LICENSE](LICENSE) file for details.

## Related Projects

- [Eventdisplay-ML](https://github.com/Eventdisplay/Eventdisplay-ML) - Machine Learning for Eventdisplay
