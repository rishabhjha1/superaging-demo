"""Export the trained ViT-B/16 SuperAger classifier for the browser demo.

Run this where the checkpoint lives (Kaggle). It writes four things into
models/:

    vit_superager.int8.onnx   quantised graph, logits + per-layer attention
    manifest.json             preprocessing constants the browser needs
    parity.json               PyTorch vs ONNX agreement, per subject
    (parity.json is summarised into manifest.parity_max_delta)

WHY THE WRAPPER EXISTS. torchvision's ViT calls nn.MultiheadAttention with
need_weights=False, so the attention matrices are never materialised in the
graph and attention rollout has nothing to consume. The wrapper below
reimplements the block forward with the same weights and emits head-averaged
attention as a second output. It is numerically identical to the original
block; test_wrapper_matches_torchvision() asserts that before anything is
exported.

DO NOT SKIP THE PARITY CHECK. Dynamic int8 quantisation moves transformer
logits. If max |delta p| exceeds ~0.02 the demo will show reviewers numbers that
disagree with Table 1, which is worse than shipping no demo at all. The number
is written into the manifest and displayed in the page footer.
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


# --------------------------------------------------------------------------
# EDIT THIS. Point it at your checkpoint and rebuild the model exactly as the
# training script does.
# --------------------------------------------------------------------------
def load_trained_model(checkpoint_path: str):
    """Return (backbone, head) with trained weights loaded, in eval mode."""
    import torchvision

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    backbone = torchvision.models.vit_b_16(weights=None)
    backbone.heads = torch.nn.Identity()

    # The frozen backbone should match the ImageNet weights the model was
    # trained on top of; load them if the checkpoint only stores the head.
    backbone_state = {k.replace("backbone.", ""): v
                      for k, v in state.items() if k.startswith("backbone.")}
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=False)
    else:
        pre = torchvision.models.vit_b_16(
            weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        backbone.load_state_dict(
            {k: v for k, v in pre.state_dict().items() if not k.startswith("heads.")},
            strict=False)

    head_state = {k.replace("head.", ""): v
                  for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise SystemExit(
            "No 'head.*' keys in the checkpoint. Adjust load_trained_model() to "
            "match how your training script names its parameters.")

    head = build_head_from_state(head_state)
    head.load_state_dict(head_state)

    return backbone.eval(), head.eval()


def build_head_from_state(state: dict) -> torch.nn.Sequential:
    """Reconstruct the classification head from its state dict keys.

    Handles the LayerNorm / Dropout / Linear / activation stack described in
    Sec. 3.3. If your head has a shape this cannot infer, replace this function
    with the literal nn.Sequential from your training script.
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
            raise SystemExit(f"Cannot infer layer {i} from weight shape {tuple(w.shape)}.")
    return torch.nn.Sequential(*layers)


class ViTWithAttention(torch.nn.Module):
    """Trained model, re-expressed so attention matrices leave the graph."""

    def __init__(self, vit, head):
        super().__init__()
        self.vit = vit
        self.head = head

    @staticmethod
    def _block(blk, x_in):
        x = blk.ln_1(x_in)
        b, n, c = x.shape
        mha = blk.self_attention
        h = mha.num_heads

        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = lambda t: t.view(b, n, h, c // h).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)

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

        feats = v.encoder.ln(x)[:, 0]           # (3, 768) per-plane CLS
        logits = self.head(feats.mean(0, keepdim=True))   # mean fusion, Sec. 3.3
        return logits, torch.stack(attns, 0)


def test_wrapper_matches_torchvision(backbone, head, tol=1e-4) -> float:
    """Assert the rewritten block forward reproduces the original model."""
    x = torch.randn(3, 3, 224, 224)
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
                 superager_index: int = 1) -> dict:
    """Compare PyTorch and quantised-ONNX SuperAger probabilities."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    deltas, rows = [], []
    for i, t in enumerate(tensors):
        with torch.no_grad():
            logits, _ = wrapped(torch.from_numpy(t))
        p_torch = torch.softmax(logits, dim=-1)[0, superager_index].item()

        out = sess.run(None, {name: t})[0]
        e = np.exp(out - out.max())
        p_onnx = float((e / e.sum())[0, superager_index])

        deltas.append(abs(p_torch - p_onnx))
        rows.append(dict(subject=i, p_torch=p_torch, p_onnx=p_onnx,
                         delta=abs(p_torch - p_onnx)))

    return dict(max_delta=float(max(deltas)), mean_delta=float(np.mean(deltas)),
                n=len(deltas), rows=rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models")
    ap.add_argument("--parity-tensors", default=None,
                    help="Optional .npy of shape (N, 3, 3, 224, 224): preprocessed "
                         "held-out subjects. Random noise is used if omitted, "
                         "which checks the graph but not the data distribution.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--site-mean", type=float, default=0.0)
    ap.add_argument("--site-sd", type=float, default=1.0)
    ap.add_argument("--imagenet-norm", action="store_true",
                    help="Set this if training applied ImageNet mean/std to the "
                         "slices. If unset the manifest uses identity, which "
                         "MUST match your training transform.")
    # Jupyter injects kernel arguments into argv; parse an empty list there.
    args = ap.parse_args([] if "ipykernel" in sys.modules else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading the checkpoint")
    backbone, head = load_trained_model(args.checkpoint)

    print("Verifying the attention wrapper")
    test_wrapper_matches_torchvision(backbone, head)

    wrapped = ViTWithAttention(backbone, head).eval()

    fp32 = out / "vit_superager.onnx"
    int8 = out / "vit_superager.int8.onnx"

    print("Exporting to ONNX")
    torch.onnx.export(
        wrapped, torch.randn(3, 3, 224, 224), str(fp32),
        input_names=["views"], output_names=["logits", "attention"],
        opset_version=17, do_constant_folding=True,
    )

    print("Quantising to int8")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    size_mb = int8.stat().st_size / 1e6
    print(f"  {int8.name} is {size_mb:.1f} MB")
    if size_mb > 95:
        print("  WARNING: GitHub rejects files over 100 MB. Split the file and "
              "concatenate the ArrayBuffers in the browser, or export fp16.")

    print("Checking parity")
    if args.parity_tensors:
        tensors = [t.astype(np.float32) for t in np.load(args.parity_tensors)]
    else:
        print("  no --parity-tensors given; using random inputs")
        tensors = [np.random.randn(3, 3, 224, 224).astype(np.float32) for _ in range(8)]

    parity = check_parity(wrapped, int8, tensors)
    print(f"  max |delta p| = {parity['max_delta']:.4f} over {parity['n']} inputs")
    if parity["max_delta"] > 0.02:
        print("  WARNING: quantisation has moved the probabilities. Ship the "
              "fp32 graph instead, or record this number prominently.")

    (out / "parity.json").write_text(json.dumps(parity, indent=2))

    manifest = dict(
        model_file=int8.name,
        exported_at=date.today().isoformat(),
        input_size=224,
        mask_fraction=0.15,
        tta=True,
        threshold=args.threshold,
        superager_index=1,
        channel_mean=[0.485, 0.456, 0.406] if args.imagenet_norm else [0.0, 0.0, 0.0],
        channel_std=[0.229, 0.224, 0.225] if args.imagenet_norm else [1.0, 1.0, 1.0],
        site_stats=dict(pooled=dict(mean=args.site_mean, sd=args.site_sd)),
        parity_max_delta=round(parity["max_delta"], 4),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {int8.name}, manifest.json and parity.json into {out}/")
    print("Delete vit_superager.onnx before committing; only the int8 graph ships.")


if __name__ == "__main__":
    main()
