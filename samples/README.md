# Sample scan

`index.html` looks for `samples/sample-t1.nii.gz`. It is not committed, because
choosing it is a licensing decision rather than a technical one.

**Do not use an ADNI scan.** The ADNI Data Use Agreement prohibits
redistribution. Publishing one here would be a breach, and a public one.
OASIS has its own terms and is also not redistributable this way.

Two options that are:

- **IXI**, released under CC BY-SA. Healthy adult T1-weighted volumes.
  <https://brain-development.org/ixi-dataset/>
- **OpenNeuro**, where many datasets are CC0. Pick a T1w NIfTI from any
  CC0-licensed dataset. <https://openneuro.org>

Whichever you pick, add a line here naming the source and its licence, and
rename the file to `sample-t1.nii.gz`.

A caveat worth stating on the page if you use IXI: those subjects are not from
the training cohort and have no SuperAger label. The sample demonstrates that
the pipeline runs; its prediction is not a validated result.
