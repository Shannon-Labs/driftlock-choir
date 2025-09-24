PDF=main.pdf

all: $(PDF)

$(PDF): main.tex refs.bib
	latexmk -pdf -quiet main.tex

clean:
	latexmk -C
	rm -f $(PDF)

.PHONY: all clean

