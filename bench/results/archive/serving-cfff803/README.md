# Serving comparison before the warm-up fix (commit cfff803)

The fix round's first serving comparison: `python bench/serve_compare.py --runs 3`, run from a
clean worktree of commit `cfff803` (see each file's `provenance`), 3 settings (`off`, `fixed`,
`adaptive`) x 4 concurrency levels x 3 runs, 5 rejected (contaminated) attempts kept as
`*.attempt<k>.json`. `serving.md` is the summary.

At that commit the service warmed up only batch sizes 1 and 8. At concurrency 8 the batcher forms
batches of 2-7, and the first call at each of those sizes was slow (the slowest 1% of requests
at concurrency 8 spent 283-538 ms in inference at a mean batch of 4.1-7.0, against 217-287 ms
for full batches of 8 at concurrency 16). `bench/batch_shape_probe.py`
(`../../serving/batch_shape_probe.json`) measured the first call at a new size at 2.7-5.9x the
repeated calls; commit `c6e3f1b` warms up every size, and `../../serving/` holds the comparison
re-run at that commit. These files are kept as the "before" of that fix.

`serving.json` and `serving.md` were first aggregated while archiving, at commit `cc76363` with
a dirty tree (`bench/profile_ops.py` modified, not the serving code). They were re-aggregated
from the same raw files with `python bench/serve_compare.py --aggregate-only --out <copy of this
directory>` at commit `8d47bd4` (clean, recorded in `aggregated_by`); every row and the table
are identical to the earlier aggregation.
