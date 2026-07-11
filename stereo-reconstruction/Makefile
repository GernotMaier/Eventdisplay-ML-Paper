# Makefile for compiling LaTeX documents
MAIN = main
LATEX = pdflatex
BIBTEX = bibtex
PDF = $(MAIN).pdf

.PHONY: all
all: $(PDF)

$(PDF): $(MAIN).tex ../bibliography/bibliography.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

# Clean auxiliary files
.PHONY: clean
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.pdf

# Clean all generated files including PDF
.PHONY: distclean
distclean: clean
	rm -f $(PDF)

.PHONY: help
help:
	@echo "Makefile for LaTeX document compilation"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Compile the LaTeX document (default)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDF"
