# Data

The two public data streams are not redistributed here. Place the following
files in this directory before running the scripts.

| File | Source |
|---|---|
| `powersupply.arff` | Stream Data Mining Repository (X. Zhu). The original address `http://www.cse.fau.edu/~xqzhu/stream.html` no longer serves the file; an archived copy is available at <https://web.archive.org/web/20240412182658/https://www.cse.fau.edu/~xqzhu/Stream/powersupply.arff>. The stream is also distributed by the USP Data Stream Repository, <https://sites.google.com/view/uspdsrepository>. |
| `elec2_data.dat`, `elec2_label.dat` | <https://github.com/vlosing/driftDatasets>, directory `realWorld/Elec2` |

The published Powersupply file contains one data line with a trailing comma and
one corrupted hour label; both are handled in `code/datasets.py`.
