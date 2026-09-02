# SuperAger classifier — browser explainability demo

A static page that runs the ViT-B/16 SuperAger classifier on a T1-weighted MRI
volume and draws the attention rollout and grid-based regional analysis from
Figure 2 of the paper. Everything runs in the visitor's browser. No scan leaves
the machine, and there is no server that could receive one.

Companion to [rishabhjha1/superaging](https://github.com/rishabhjha1/superaging),
which remains the reproducible artifact for the paper. This repository is a
convenience for reviewers who want to poke at the model interactively.

---

## What ships and what does not

| Method | Here | Why |
|---|---|---|
| Attention rollout | yes | Forward-only. Ports exactly. |
| Grid-based regional analysis | yes | Verified against `regions.py` to 4.5e-7. |
| Grad-CAM | no | See below. |
| LIME | no | See below. |

**Grad-CAM.** It needs gradients with respect to patch tokens. With a frozen
backbone, only the class token reaches the classification head, so patch-token
gradients are identically zero unless you backpropagate through the final
encoder block. ONNX Runtime Web has no autograd, so shipping it would mean a
hand-written backward pass — a new and untestable source of disagreement with
the published figures. It stays in the notebook.

**LIME.** 1000 masking perturbations is 1000 forward passes. Single-threaded
WebAssembly puts that around ten minutes. Cutting to 200 would move the local
fidelity away from the 0.78 reported in Sec. 4.2. It stays in the notebook.

The page says all of this on screen, so a reviewer is never left wondering
whether a panel is missing or broken.

---

## Deploying

### 1. Export the model

Run this where the checkpoint lives, on Kaggle:

```bash
python scripts/export_onnx.py \
  --checkpoint /kaggle/input/…/vit_best.pt \
  --out models \
  --threshold 0.5 \
  --parity-tensors /kaggle/working/heldout_tensors.npy
```

`--parity-tensors` takes an `(N, 3, 3, 224, 224)` array of preprocessed held-out
subjects. Without it the script falls back to random noise, which checks that
the graph is wired correctly but tells you nothing about how quantisation
behaves on real scans. Provide it.

Two flags decide whether the demo is correct:

- `--imagenet-norm` if your training transform applied ImageNet mean and std to
  the slices. If this does not match training, the page runs cleanly and returns
  wrong probabilities.
- `--site-mean` / `--site-sd`, the pooled training-cohort statistics used in
  place of per-site normalisation.

The script refuses to export if its attention wrapper does not reproduce your
original model, and warns if int8 quantisation has moved probabilities by more
than 0.02. Read both messages before committing.

### 2. Add a sample scan

See `samples/README.md`. Do not use ADNI or OASIS; the DUAs prohibit
redistribution. IXI (CC BY-SA) or a CC0 OpenNeuro volume both work.

### 3. Push and turn on Pages

```bash
gh repo create rishabhjha1/superaging-demo --public --source=. --push
```

Then, in the repository: **Settings → Pages → Source: Deploy from a branch →
`main` / `(root)` → Save.**

Live at `https://rishabhjha1.github.io/superaging-demo/` in about a minute.
Every later `git push` redeploys. No Actions workflow is needed and adding one
buys nothing here.

### 4. Check it before sending the link

Open the page in a private window on a machine that has never seen it, and watch
the network tab. If the ONNX request returns a couple of hundred bytes of text
beginning `version https://git-lfs`, the model got committed through LFS and
Pages is serving the pointer. Fix that first.

---

## Four things about GitHub Pages that will bite you

**Git LFS breaks the model.** Pages serves LFS files as pointer text, not
binaries. Commit `vit_superager.int8.onnx` as an ordinary file. GitHub's hard
limit is 100 MB per file and an int8 ViT-B/16 lands near 87 MB, so run
`ls -lh models/` first.

**No custom headers, so no threads.** Pages cannot set the COOP and COEP headers
that `SharedArrayBuffer` requires, so ONNX Runtime's WASM backend is
single-threaded: roughly 3–8 s per forward pass, near 40 s for the four TTA
passes. WebGPU is tried first and is much faster where available. If you want
threads anyway, `coi-serviceworker` shims the headers from a service worker and
does work on Pages — keep the model same-origin if you add it, since COEP will
otherwise block cross-origin fetches.

**`.nojekyll` must exist.** Without it Jekyll processes the site and silently
drops anything under a directory beginning with `_`. It is already in this
repository; do not delete it.

**Bandwidth.** The 100 GB/month soft limit divided by 87 MB is about 1,100 cold
loads. Ample for a review cycle.

---

## Correctness guards built in

**Orientation.** `DEFAULT_GRID` assumes canonical RAS. A transposed or flipped
volume swaps frontal and parietal labels while every number in the table stays
plausible — the exact failure `scripts/check_grid_orientation.py` exists to
catch in the main repository. Here it runs automatically on every upload, and
the regional table is withheld rather than shown wrong. The attention panels,
which do not depend on the grid, are still displayed.

**Site normalisation.** An uploaded scan has no site label, so per-site
z-scoring has nothing to key on. The pooled training-cohort statistics from the
manifest are applied instead, and the deviation is documented in the header of
`lib/preprocess.js` rather than hidden.

**Peak scaling.** `I_R` is reported relative to the peak region, matching
`regions.py`. The page states this above the table, since it is not
self-evident.

---

## Known deviations from the Python pipeline

Stated here so they are on the record rather than discovered by a reviewer.

1. Histogram matching against the n = 20 reference is not implemented. It is a
   refinement on an already-robust normalisation.
2. The consistency sweep varies the anatomical slice the grid is applied to, but
   computes attention once at the mid-slice. Running the full model at five
   depths would be five times the inference cost, which single-threaded WASM
   cannot absorb.
3. int8 dynamic quantisation shifts logits slightly. The measured shift is in
   `models/parity.json` and printed in the page footer.

---

## Layout

```
index.html              page and styles
app.js                  orchestration
lib/preprocess.js       Sec. 3.2 pipeline: RAS, mask, normalise, 2.5D slices
lib/model.js            ONNX Runtime session, TTA
lib/rollout.js          attention rollout
lib/regions.js          port of regions.py, verified to 4.5e-7
lib/orientation.js      RAS guard
lib/render.js           canvas panels
scripts/export_onnx.py  checkpoint -> ONNX + manifest + parity report
models/                 exported artifacts (see models/README.md)
samples/                sample scan (see samples/README.md)
```

No build step, no bundler, no `npm install`. ONNX Runtime and nifti-reader-js
load from cdnjs.

---

## Before you put the URL in the manuscript

Check whether MICAD review is double-blind. A `github.io/rishabhjha1` link would
deanonymize the submission.
