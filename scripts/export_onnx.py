"""Export the trained ViT-B/16 SuperAger classifier for the browser demo.

Run this where the checkpoint lives (Kaggle). It writes into models/:

    vit_superager.int8.onnx   quantised graph, logits + per-layer attention
    manifest.json             the input contract and preprocessing constants
    parity.json               PyTorch vs ONNX agreement, fp32 and int8

WHY THE WRAPPER EXISTS. torchvision's ViT calls nn.MultiheadAttention with
need_weights=False, so attention matrices never enter the graph and attention
rollout has nothing to consume. The wrapper reimplements the block forward with
the same weights and emits head-averaged attention as a second output. It is
numerically identical to the original block; test_wrapper_matches_torchvision()
asserts that before anything is exported.

INPUT IS ONE SUBJECT, NOT A BATCH. The graph takes (3, 3, 224, 224): three
planes, each a 3-channel image. Mean fusion (Sec. 3.3) averages over axis 0, so
a dynamic batch axis there would silently average across *subjects* instead of
across planes and produce plausible, wrong probabilities. Test-time
augmentation is therefore looped, not batched.

DO NOT SKIP THE PARITY CHECK. It runs twice: fp32 ONNX against PyTorch (should
be ~1e-6; anything larger means the export itself is broken) and int8 against
PyTorch (dynamic quantisation moves transformer logits). Both numbers go into
the manifest so the page footer can show them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

INPUT_SHAPE = (3, 3, 224, 224)
CLASS_NAMES = ["Typical", "SuperAger"]


# --------------------------------------------------------------------------
# EDIT THIS if your training script names parameters differently.
# --------------------------------------------------------------------------
def load_trained_model(checkpoint_path: str | None, allow_untrained: bool = False):
    """Return (backbone, head) in eval mode.

    With allow_untrained and no checkpoint, the backbone gets ImageNet weights
    (or random weights if they cannot be downloaded) and the head is randomly
    initialised. That exercises every step below -- wrapper check, export,
    quantisation, parity -- without a checkpoint, so plumbing failures surface
    before you go looking for one. It must never produce a shipped model.
    """
    import torchvision

    if checkpoint_path is None:
        if not allow_untrained:
            raise SystemExit("--checkpoint is required (or pass --allow-untrained "
                             "to validate the pipeline with random weights).")
        print("  UNTRAINED MODE: random head, no checkpoint. Not for release.")
        backbone = _imagenet_backbone(torchvision)
        head = torch.nn.Sequential(
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(768, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(256, 2),
        )
        return backbone.eval(), head.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    backbone = torchvision.models.vit_b_16(weights=None)
    backbone.heads = torch.nn.Identity()

    backbone_state = {k.replace("backbone.", ""): v
                      for k, v in state.items() if k.startswith("backbone.")}
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=False)
    else:
        backbone = _imagenet_backbone(torchvision)

    head_state = {k.replace("head.", ""): v
                  for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise SystemExit(
            f"No 'head.*' keys in the checkpoint. Top-level keys are: "
            f"{sorted(state)[:12]}. Adjust load_trained_model() to match how "
            f"your training script names its parameters.")

    head = build_head_from_state(head_state)
    head.load_state_dict(head_state)
    return backbone.eval(), head.eval()


def _imagenet_backbone(torchvision):
    """ViT-B/16 with ImageNet weights, falling back to random if offline."""
    backbone = torchvision.models.vit_b_16(weights=None)
    backbone.heads = torch.nn.Identity()
    try:
        pre = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        backbone.load_state_dict(
            {k: v for k, v in pre.state_dict().items() if not k.startswith("heads.")},
            strict=False)
    except Exception as exc:  # offline Kaggle kernel, no cached weights
        print(f"  could not fetch ImageNet weights ({type(exc).__name__}); "
              f"backbone is randomly initialised")
    return backbone


def build_head_from_state(state: dict) -> torch.nn.Sequential:
    """Reconstruct the classification head from its state dict keys.

    Handles the LayerNorm / Dropout / Linear / activation stack of Sec. 3.3.
    If your head has a shape this cannot infer, replace this function with the
    literal nn.Sequential from your training script.
    """
    indices = sorted({int(k.split(".")[0]) for k in state if k.split(".")[0].isdigit()})
    if not indices:
        raise SystemExit("Head state dict is not indexed like an nn.Sequential; "
                         "replace build_head_from_state() with your head definition.")

    layers: list[torch.nn.Module] = []
    for i in range(max(indices) + 1):
        w = state.get(f"{i}.weight")
        b = state.get(f"{i}.bias")
        if w is None:
            layers.append(torch.nn.GELU())
        elif w.ndim == 1:
            layers.append(torch.nn.LayerNorm(w.shape[0]))
        elif w.ndim == 2:
            layers.append(torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None))
        else:
            raise SystemExit(
                f"Cannot infer layer {i} from weight shape {tuple(w.shape)}.")
    return torch.nn.Sequential(*layers)


class ViTWithAttention(torch.nn.Module):
    """Trained model, re-expressed so attention matrices leave the graph."""

    def __init__(self, vit, head):
        super().__init__()
        self.vit = vit
        self.head = head

    @staticmethod
    def _split_heads(t: torch.Tensor, b: int, n: int, h: int, c: int) -> torch.Tensor:
        return t.view(b, n, h, c // h).transpose(1, 2)

    def _block(self, blk, x_in):
        x = blk.ln_1(x_in)
        b, n, c = x.shape
        mha = blk.self_attention
        h = mha.num_heads

        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q, b, n, h, c)
        k = self._split_heads(k, b, n, h, c)
        v = self._split_heads(v, b, n, h, c)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(c // h)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = mha.out_proj(out) + x_in
        return x + blk.mlp(blk.ln_2(x)), attn.mean(dim=1)

    def forward(self, views):
        """views: (3, 3, 224, 224), one 3-channel image per plane."""
        v = self.vit
        x = v._process_input(views)
        cls = v.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + v.encoder.pos_embedding

        attns = []
        for blk in v.encoder.layers:
            x, a = self._block(blk, x)
            attns.append(a)

        feats = v.encoder.ln(x)[:, 0]                     # (3, 768) per-plane CLS
        logits = self.head(feats.mean(0, keepdim=True))   # mean fusion, Sec. 3.3
        return logits, torch.stack(attns, 0)


# --------------------------------------------------------------------------
# Test-time augmentation (Sec. 3.4): original, horizontal flip, x1.05, x0.95.
# Looped rather than batched -- see the module docstring.
# --------------------------------------------------------------------------
def tta_views(x: torch.Tensor) -> list[torch.Tensor]:
    return [x, torch.flip(x, dims=[-1]), _rescale(x, 1.05), _rescale(x, 0.95)]


def _rescale(x: torch.Tensor, factor: float) -> torch.Tensor:
    """Zoom about the centre, cropping or padding back to 224x224."""
    size = x.shape[-1]
    scaled = F.interpolate(x, scale_factor=factor, mode="bilinear",
                           align_corners=False, recompute_scale_factor=False)
    new = scaled.shape[-1]
    if new >= size:
        off = (new - size) // 2
        return scaled[..., off:off + size, off:off + size]
    pad = size - new
    lo = pad // 2
    return F.pad(scaled, (lo, pad - lo, lo, pad - lo))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def test_wrapper_matches_torchvision(backbone, head, tol=1e-4) -> float:
    """Assert the rewritten block forward reproduces the original model."""
    x = torch.randn(*INPUT_SHAPE)
    wrapped = ViTWithAttention(backbone, head).eval()

    with torch.no_grad():
        mine, _ = wrapped(x)
        feats = backbone(x)
        theirs = head(feats.mean(0, keepdim=True))

    delta = (mine - theirs).abs().max().item()
    if delta > tol:
        raise SystemExit(
            f"The attention wrapper does not reproduce the original model "
            f"(max |delta| = {delta:.2e} > {tol:.0e}). Do not export. Check that "
            f"load_trained_model() rebuilds the architecture you trained.")
    print(f"  wrapper matches torchvision, max |delta| = {delta:.2e}")
    return delta


def check_parity(wrapped, onnx_path: Path, tensors: list[np.ndarray],
                 superager_index: int = 1, use_tta: bool = True) -> dict:
    """Compare PyTorch and ONNX SuperAger probabilities, TTA included.

    The demo shows a TTA-averaged probability, so the parity number has to
    average the same four views. Otherwise it validates something reviewers
    never see.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    deltas, rows = [], []
    for i, t in enumerate(tensors):
        base = torch.from_numpy(t)
        views = tta_views(base) if use_tta else [base]

        p_torch, p_onnx = [], []
        for view in views:
            with torch.no_grad():
                logits, _ = wrapped(view)
            p_torch.append(torch.softmax(logits, dim=-1)[0, superager_index].item())

            out = sess.run(None, {name: view.numpy().astype(np.float32)})[0]
            p_onnx.append(float(_softmax_np(out)[0, superager_index]))

        pt, po = float(np.mean(p_torch)), float(np.mean(p_onnx))
        deltas.append(abs(pt - po))
        rows.append(dict(subject=i, p_torch=pt, p_onnx=po, delta=abs(pt - po)))

    return dict(max_delta=float(max(deltas)), mean_delta=float(np.mean(deltas)),
                n=len(deltas), tta=use_tta, rows=rows)


def load_parity_tensors(path: str | None, n_random: int = 8) -> list[np.ndarray]:
    """Load preprocessed subjects, accepting either cache layout.

    The preprocessed cache is (N, 3, 224, 224): one 3-plane stack per subject,
    single-channel planes. The graph wants (3, 3, 224, 224) per subject, so each
    plane is repeated across the channel axis -- the same grayscale-to-RGB step
    the training loader does. A (N, 3, 3, 224, 224) array is used as-is.
    """
    if path is None:
        print("  no --parity-tensors given; using random inputs, which checks "
              "the graph but not the data distribution")
        return [np.random.randn(*INPUT_SHAPE).astype(np.float32)
                for _ in range(n_random)]

    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[1:] == (3, 224, 224):
        print(f"  {path}: {arr.shape} -> repeating planes across channels")
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 5 and arr.shape[1:] == INPUT_SHAPE:
        print(f"  {path}: {arr.shape}, used as-is")
    else:
        raise SystemExit(
            f"--parity-tensors has shape {arr.shape}; expected (N, 3, 224, 224) "
            f"or (N, 3, 3, 224, 224).")

    lo, hi = float(arr.min()), float(arr.max())
    print(f"  value range [{lo:.3f}, {hi:.3f}] -- confirm this matches what the "
          f"browser produces after preprocessing")
    return [t.astype(np.float32) for t in arr]


def export_graph(wrapped, path: Path) -> None:
    """Export at fixed (3, 3, 224, 224). No dynamic batch axis: see docstring."""
    kwargs = dict(
        input_names=["views"], output_names=["logits", "attention"],
        opset_version=17, do_constant_folding=True,
    )
    # torch >= 2.9 defaults to the dynamo exporter; pin the legacy path so the
    # graph is the same whatever torch the kernel happens to have.
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(wrapped, torch.randn(*INPUT_SHAPE), str(path), **kwargs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--allow-untrained", action="store_true",
                    help="Run the whole pipeline with a random head, to check "
                         "plumbing before the checkpoint is available.")
    ap.add_argument("--out", default="models")
    ap.add_argument("--parity-tensors", default=None,
                    help="Optional .npy, (N, 3, 224, 224) or (N, 3, 3, 224, 224).")
    ap.add_argument("--parity-subjects", type=int, default=8,
                    help="Cap on how many subjects to run parity over.")
    ap.add_argument("--no-tta", action="store_true",
                    help="Skip TTA in the parity check. Only if the demo also "
                         "skips it; the manifest records which was used.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Operating threshold from the validation pool (Sec. 3.4), "
                         "not necessarily 0.5.")
    ap.add_argument("--site-mean", type=float, default=0.0)
    ap.add_argument("--site-sd", type=float, default=1.0)
    ap.add_argument("--imagenet-norm", action="store_true",
                    help="Set only if training applied ImageNet mean/std.")
    ap.add_argument("--keep-fp32", action="store_true",
                    help="Keep the fp32 graph in models/ (it is deleted by "
                         "default; only the int8 graph ships).")
    # Jupyter injects kernel arguments into argv; parse an empty list there.
    if argv is None:
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading the model")
    backbone, head = load_trained_model(args.checkpoint, args.allow_untrained)

    print("Verifying the attention wrapper")
    test_wrapper_matches_torchvision(backbone, head)
    wrapped = ViTWithAttention(backbone, head).eval()

    fp32 = out / "vit_superager.onnx"
    int8 = out / "vit_superager.int8.onnx"

    print("Exporting to ONNX")
    export_graph(wrapped, fp32)

    print("Quantising to int8")
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    size_mb = int8.stat().st_size / 1e6
    print(f"  {int8.name} is {size_mb:.1f} MB")
    if size_mb > 95:
        print("  WARNING: GitHub rejects files over 100 MB. Split the file and "
              "concatenate the ArrayBuffers in the browser, or export fp16.")

    print("Checking parity")
    tensors = load_parity_tensors(args.parity_tensors)[:args.parity_subjects]
    use_tta = not args.no_tta

    fp32_parity = check_parity(wrapped, fp32, tensors, use_tta=use_tta)
    print(f"  fp32 vs torch: max |delta p| = {fp32_parity['max_delta']:.2e}")
    if fp32_parity["max_delta"] > 1e-3:
        print("  WARNING: the fp32 graph already disagrees with PyTorch, so the "
              "export itself is wrong. Fix that before reading the int8 number.")

    int8_parity = check_parity(wrapped, int8, tensors, use_tta=use_tta)
    print(f"  int8 vs torch: max |delta p| = {int8_parity['max_delta']:.4f} "
          f"over {int8_parity['n']} subjects")
    if int8_parity["max_delta"] > 0.02:
        print("  WARNING: quantisation has moved the probabilities. Ship the "
              "fp32 graph instead, or record this number prominently.")

    (out / "parity.json").write_text(json.dumps(
        dict(fp32=fp32_parity, int8=int8_parity), indent=2))

    manifest = dict(
        model_file=int8.name,
        exported_at=date.today().isoformat(),
        untrained=args.checkpoint is None,
        # Input contract: the frontend should fail loudly at load if it cannot
        # build exactly this, rather than at session.run with a shape error.
        input_name="views",
        input_shape=list(INPUT_SHAPE),
        output_names=["logits", "attention"],
        attention_shape=[12, 3, 197, 197],
        class_names=CLASS_NAMES,
        superager_index=1,
        input_size=224,
        mask_fraction=0.15,
        tta=use_tta,
        tta_views=["original", "hflip", "scale_1.05", "scale_0.95"] if use_tta else [],
        threshold=args.threshold,
        channel_mean=[0.485, 0.456, 0.406] if args.imagenet_norm else [0.0, 0.0, 0.0],
        channel_std=[0.229, 0.224, 0.225] if args.imagenet_norm else [1.0, 1.0, 1.0],
        site_stats=dict(pooled=dict(mean=args.site_mean, sd=args.site_sd)),
        parity_max_delta=round(int8_parity["max_delta"], 4),
        parity_fp32_max_delta=float(f"{fp32_parity['max_delta']:.3e}"),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.keep_fp32:
        fp32.unlink(missing_ok=True)

    print(f"\nWrote {int8.name}, manifest.json and parity.json into {out}/")
    if args.checkpoint is None:
        print("UNTRAINED: manifest.untrained is true. Do not commit this model.")


if __name__ == "__main__":
    main()
