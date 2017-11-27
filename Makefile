FNAME_BASE=COSC-6365-MichaelYantosca-FinalProjectProposal

all: docs

docs: $(FNAME_BASE).pdf

$(FNAME_BASE).pdf: $(FNAME_BASE).tex
	@lualatex -shell-escape $(FNAME_BASE).tex
	@biber $(FNAME_BASE)
	@lualatex -shell-escape $(FNAME_BASE).tex

superclean: clean
	rm -f $(FNAME_BASE).pdf

clean:
	@rm -f $(FNAME_BASE).aux
	@rm -f $(FNAME_BASE).bbl
	@rm -f $(FNAME_BASE).bcf
	@rm -f $(FNAME_BASE).log
	@rm -f $(FNAME_BASE).run.xml
	@rm -f $(FNAME_BASE).dvi
	@rm -f $(FNAME_BASE).blg
	@rm -f $(FNAME_BASE).auxlock
	@rm -f $(FNAME_BASE).pyg
	@rm -f $(FNAME_BASE)-figure*
	@rm -f $(FNAME_BASE).toc
	@rm -f *~
