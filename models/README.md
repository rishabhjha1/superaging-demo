# Model files

Three files belong here. None is committed yet, because they are produced from
your trained checkpoint rather than from this repository.

| File | Produced by | Committed? |
|---|---|---|
| `vit_superager.int8.onnx` | `scripts/export_onnx.py` | yes, as a plain file |
| `manifest.json` | `scripts/export_onnx.py` | yes |
| `parity.json` | `scripts/export_onnx.py` | yes |

The unquantised `vit_superager.onnx` is an intermediate. It is in `.gitignore`;
do not commit it.

## Do not use Git LFS

GitHub Pages serves LFS-tracked files as their pointer text rather than the
binary, so the model would arrive as a few dozen bytes of metadata and the
session would fail to start. Commit the ONNX as an ordinary file.

GitHub's hard per-file limit is 100 MB. An int8-quantised ViT-B/16 lands near
87 MB, so check `ls -lh models/` before committing. If it is over, either export
fp16 instead, or split the file and concatenate the ArrayBuffers in
`lib/model.js`.

## What manifest.json controls

`channel_mean` and `channel_std` must match the normalisation your training
transform applied to the slices. Getting this wrong produces a page that runs
without error and returns wrong probabilities, which is the worst failure mode
available. Pass `--imagenet-norm` to the export script if training applied
ImageNet statistics.

`site_stats.pooled` holds the training-cohort mean and SD used in place of
per-site normalisation, since an uploaded scan has no site label. See the header
of `lib/preprocess.js`.
