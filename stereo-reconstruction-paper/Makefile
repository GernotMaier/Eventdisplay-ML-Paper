# Makefile for compiling LaTeX documents
# Generic LaTeX compilation with BibTeX support

# Main LaTeX file (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex

# BibTeX compiler
BIBTEX = bibtex

# Output PDF file
PDF = $(MAIN).pdf

# Default target
.PHONY: all
all: $(PDF)

# Compile the PDF
$(PDF): $(MAIN).tex bibliography.bib
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

# View the PDF (requires a PDF viewer)
.PHONY: view
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Help target
.PHONY: help
help:
	@echo "Makefile for LaTeX document compilation"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Compile the LaTeX document (default)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove all generated files including PDF"
	@echo "  view       - Open the compiled PDF"
	@echo "  help       - Show this help message"
