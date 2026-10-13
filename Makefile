LATEX ?= tectonic

.PHONY: all article dist clean

all: article

article:
	mkdir -p build
	$(LATEX) -o build article.tex

# Zip for Overleaf: class, article and figures.
dist:
	rm -rf dist
	mkdir -p dist/article
	cp misisarticle.cls article.tex LICENSE dist/article/
	cp -r figures dist/article/
	cd dist && zip -r article.zip article
	rm -rf dist/article

clean:
	rm -rf build dist
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.bcf *.run.xml *.xdv *.synctex.gz *.fls *.fdb_latexmk
