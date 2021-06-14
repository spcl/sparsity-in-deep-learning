.PHONY: all

all: README.md

README.md: sparsity-bib.tex sparsity.bib
	pandoc sparsity-bib.tex -o index.html --bibliography sparsity.bib
	node tomarkdown.js

clean:
	rm -f README.md index.html
