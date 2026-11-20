# Build with `make` (or `python3 generate.py && latexmk`). Intermediate files go to build/, the PDF stays here.
@default_files = ('report.tex');
$pdf_mode = 1;
$aux_dir = 'build';
$out_dir = '.';
$bibtex_use = 2;
