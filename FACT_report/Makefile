all: build/FACT.pdf

build/FACT.pdf: FORCE FACT.tex build/A.pdf build/flux.pdf build/theta_sq.pdf build/unfolding.pdf | build
	TEXTINPUTS="$(call translate,build:)" \
	BIBINPUTS=build: \
	max_print_line=1048576 \
	latexmk	\
		--lualatex \
		--output-directory=build	\
		--interaction=nonstopmode	\
		--halt-on-error	\
	FACT.tex

build/theta_sq.pdf: FACT.py data/gamma_corsika_headers.hdf5 data/gamma_test_dl3.hdf5 data/open_crab_sample_dl3.hdf5 |build

	TEXTINPUTS=$$(pwd): python FACT.py

build/A.pdf: FACT.py data/gamma_corsika_headers.hdf5 data/gamma_test_dl3.hdf5 data/open_crab_sample_dl3.hdf5 |build

	TEXTINPUTS=$$(pwd): python FACT.py

build/unfolding.pdf: FACT.py data/gamma_corsika_headers.hdf5 data/gamma_test_dl3.hdf5 data/open_crab_sample_dl3.hdf5 |build

	TEXTINPUTS=$$(pwd): python FACT.py

build/flux.pdf: FACT.py data/gamma_corsika_headers.hdf5 data/gamma_test_dl3.hdf5 data/open_crab_sample_dl3.hdf5 |build

	TEXTINPUTS=$$(pwd): python FACT.py


build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: FORCE all clean
	