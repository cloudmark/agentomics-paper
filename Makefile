.PHONY: all clean

# generated.tex holds every number, table and chart series, written from data/report-data.json.
all: generated.tex
	latexmk

generated.tex: generate.py data/report-data.json
	python3 generate.py

clean:
	latexmk -C
	rm -rf build generated.tex
