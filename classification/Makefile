# Makefile for compiling the classification note

MAIN = main
LATEX = pdflatex
PDF = $(MAIN).pdf

.PHONY: all
all: $(PDF)

$(PDF): $(MAIN).tex ../stereo-reconstruction-paper/aas_macros.sty
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

.PHONY: clean
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz *.pdf

.PHONY: distclean
distclean: clean
	rm -f $(PDF)

.PHONY: view
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

.PHONY: help
help:
	@echo "Makefile for LaTeX note compilation"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Compile the LaTeX document (default)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDF"
	@echo "  view       - Open the compiled PDF"
	@echo "  help       - Show this help message"
