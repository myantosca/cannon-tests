PROPOSAL_FNAME_BASE=COSC-6365-MichaelYantosca-FinalProjectProposal
REPORT_FNAME_BASE=COSC-6365-MichaelYantosca-FinalProjectReport
SLIDES_FNAME_BASE=COSC-6365-MichaelYantosca-FinalProjectSlides
all: docs exe

exe: matrixMulCUBLAS mkl_cblas_sgemm cannon

docs: $(PROPOSAL_FNAME_BASE).pdf $(REPORT_FNAME_BASE).pdf $(SLIDES_FNAME_BASE).pdf

%.pdf: %.tex %.bib
	@lualatex -shell-escape $*.tex
	@biber $*
	@lualatex -shell-escape $*.tex

matrixMulCUBLAS:
	@make -C 3pty/matrixMulCUBLAS
	@cp 3pty/matrixMulCUBLAS/matrixMulCUBLAS .

mkl_cblas_sgemm:
	@make -C 3pty/mkl_cblas_sgemm
	@cp 3pty/mkl_cblas_sgemm/mkl_cblas_sgemm .

cannon:
	@make -C src
	@cp src/cannon-su .
	@cp src/cannon-so .

superclean: clean superclean-doc-$(PROPOSAL_FNAME_BASE) superclean-doc-$(REPORT_FNAME_BASE) superclean-doc-$(SLIDES_FNAME_BASE)

superclean-doc-%:
	rm -f $*.pdf

clean: clean-doc-$(PROPOSAL_FNAME_BASE) clean-doc-$(REPORT_FNAME_BASE) clean-doc-$(SLIDES_FNAME_BASE)
	@rm -f *~
	@rm -f matrixMulCUBLAS
	@rm -f mkl_cblas_sgemm
	@rm -f cannon-su
	@rm -f cannon-so
	@make -C src clean
	@make -C 3pty/matrixMulCUBLAS clean
	@make -C 3pty/mkl_cblas_sgemm clean

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
	@rm -f $*.out
	@rm -f $*.snm
	@rm -f $*.nav
