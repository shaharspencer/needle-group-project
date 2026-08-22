# Final analysis datasets

This folder contains only the four datasets used for the reported analyses and
label checks. It does not contain model scores, diagnostics, or figures.

| File | Rows | Unit of observation |
| --- | ---: | --- |
| [q1_characters.csv](q1_characters.csv) | 15,686 | One named, on-screen character, with the death label and all candidate Q1 features. |
| [q2_franchises.csv](q2_franchises.csv) | 38 | One franchise, with mortality and the franchise attributes used for Q2. |
| [annotation_sample.csv](annotation_sample.csv) | 600 | One wiki page in the stratified label-checking sample. |
| [unclear_case_labels.csv](unclear_case_labels.csv) | 1,858 | One case whose infobox did not give a usable death status, with its rubric-based label. |

Checks against the writeup:

- Q1 contains 5,579 deaths, a mortality rate of 35.57%, across 38 franchises.
- Q2 summarizes 17,584 named, on-screen characters across 38 franchises.
- The unclear-case labels contain 123 dead, 1,223 alive, and 512 undecided.

Large CSVs use Git LFS. On GitHub, use the download button on the file page to
download the actual CSV rather than the small LFS pointer.
