PROPOSAL_FNAME_BASE=COSC-6365-MichaelYantosca-FinalProjectProposal
REPORT_FNAME_BASE=COSC-6365-MichaelYantosca-FinalProjectReport

all: docs

docs: $(PROPOSAL_FNAME_BASE).pdf $(REPORT_FNAME_BASE).pdf

%.pdf: %.tex %.bib
	@lualatex -shell-escape $*.tex
	@biber $*
	@lualatex -shell-escape $*.tex

superclean: clean superclean-doc-$(PROPOSAL_FNAME_BASE) superclean-doc-$(REPORT_FNAME_BASE)

superclean-doc-%:
	rm -f $*.pdf

clean: clean-doc-$(PROPOSAL_FNAME_BASE) clean-doc-$(REPORT_FNAME_BASE)
	@rm -f *~

clean-doc-%: 
	@rm -f $*.aux
	@rm -f $*.bbl
	@rm -f $*.bcf
	@rm -f $*.log
	@rm -f $*.run.xml
	@rm -f $*.dvi
	@rm -f $*.blg
	@rm -f $*.auxlock
	@rm -f $*.pyg
	@rm -f $*-figure*
	@rm -f $*.toc

