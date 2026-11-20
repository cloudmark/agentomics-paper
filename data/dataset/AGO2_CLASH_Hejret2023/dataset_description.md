# AGO2 CLASH (Hejret 2023): microRNA target sites

Binary classification over pairs of RNA sequences: does this microRNA bind this target site? The pairs come from an
AGO2 CLASH experiment, which captures a microRNA joined to the stretch of transcript it was bound to; the set is the
`AGO2_CLASH_Hejret2023` one of the miRBench benchmark.

`input/data.csv` has one row per pair; its `id` column is the sample's id in `labels.csv`:

- `gene`: the candidate target site, 50 nucleotides of the transcript, written as DNA (`A`, `C`, `G`, `T`), 5' to 3'
- `noncodingRNA`: the microRNA's sequence, 17 to 26 nucleotides (most are 22), written as DNA too, 5' to 3'

The label is `1` for a pair seen bound in the experiment and `0` for a negative pair (a microRNA with a site it was not
seen bound to). The classes are balanced. The same microRNA appears in many rows (435 distinct ones in training), with
both labels; a target site appears once.

There are no other features: what to derive from the two sequences (the match of the site to the microRNA's seed,
its nucleotides 2 to 8, base pairing along the rest, k-mers, an encoding for a sequence model) is the work.
