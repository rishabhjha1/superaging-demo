/**
 * 2.5D multi-plane preprocessing (paper Sec. 3.2), ported to the browser.
 *
 * Order of operations mirrors the Python pipeline exactly:
 *   1. reorient to canonical RAS
 *   2. strict brain mask (15%-of-max, hole fill, largest connected component)
 *   3. robust median/IQR normalisation within the mask
 *   4. per-scan [1, 99] percentile clip, rescale to [0, 1]
 *   5. per-site z-score  <-- see SITE NOTE below
 *   6. second [1, 99] percentile clip, rescale to [0, 1]
 *   7. axial / coronal / sagittal mid-slices, bilinear resize to 224 x 224
 *
 * SITE NOTE. An uploaded scan has no site label, so step 5 has nothing to key
 * on. We apply the pooled training-cohort statistics from models/manifest.json
 * and surface which statistics were used in the UI. This is a deliberate,
 * disclosed deviation: a reviewer's scan is normalised against the training
 * distribution, not against itself. Self-normalising would apply a transform
 * the model never saw and shift the prediction for reasons the UI could not
 * explain.
 *
 * Step 4's histogram matching against the n = 20 reference is NOT implemented
 * here. It is a refinement on an already-robust normalisation; omitting it is
 * recorded in the parity report rather than hidden.
 */

/** Axis codes ('R'|'L', 'A'|'P', 'S'|'I') implied by a 4x4 affine. */
export function affineToAxisCodes(affine) {
  const codes = [];
  for (let col = 0; col < 3; col++) {
    let best = 0;
    let bestAbs = -1;
    for (let row = 0; row < 3; row++) {
      const v = Math.abs(affine[row][col]);
      if (v > bestAbs) { bestAbs = v; best = row; }
    }
    const positive = affine[best][col] >= 0;
    codes.push(['RL', 'AP', 'SI'][best][positive ? 0 : 1]);
  }
  return codes;
}

/**
 * Reorient a volume to canonical RAS.
 * Returns { data, dims, codes } with axis 0 = L->R, 1 = P->A, 2 = I->S.
 */
export function reorientToRAS(data, dims, affine) {
  const codes = affineToAxisCodes(affine);
  const target = ['R', 'A', 'S'];
  const flipOf = { R: 'L', L: 'R', A: 'P', P: 'A', S: 'I', I: 'S' };

  // Which source axis carries each target direction, and does it need flipping?
  const perm = [0, 0, 0];
  const flip = [false, false, false];
  for (let t = 0; t < 3; t++) {
    for (let s = 0; s < 3; s++) {
      if (codes[s] === target[t]) { perm[t] = s; flip[t] = false; }
      else if (codes[s] === flipOf[target[t]]) { perm[t] = s; flip[t] = true; }
    }
  }

  const outDims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
  const out = new Float32Array(outDims[0] * outDims[1] * outDims[2]);
  const srcStride = [1, dims[0], dims[0] * dims[1]];

  for (let z = 0; z < outDims[2]; z++) {
    for (let y = 0; y < outDims[1]; y++) {
      for (let x = 0; x < outDims[0]; x++) {
        const idx = [x, y, z];
        let srcOffset = 0;
        for (let t = 0; t < 3; t++) {
          const s = perm[t];
          const i = flip[t] ? dims[s] - 1 - idx[t] : idx[t];
          srcOffset += i * srcStride[s];
        }
        out[x + y * outDims[0] + z * outDims[0] * outDims[1]] = data[srcOffset];
      }
    }
  }
  return { data: out, dims: outDims, codes };
}

/** Strict brain mask: threshold, fill holes, keep the largest component. */
export function brainMask(data, dims, fraction = 0.15) {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;

  let max = 0;
  for (let i = 0; i < n; i++) if (data[i] > max) max = data[i];
  const thr = max * fraction;

  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = data[i] > thr ? 1 : 0;

  const largest = largestComponent(mask, dims);
  return fillHoles(largest, dims);
}

function largestComponent(mask, dims) {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const labels = new Int32Array(n).fill(-1);
  const stack = new Int32Array(n);
  let bestLabel = -1;
  let bestSize = 0;
  let label = 0;

  for (let seed = 0; seed < n; seed++) {
    if (mask[seed] !== 1 || labels[seed] !== -1) continue;
    let top = 0;
    stack[top++] = seed;
    labels[seed] = label;
    let size = 0;

    while (top > 0) {
      const p = stack[--top];
      size++;
      const z = (p / (nx * ny)) | 0;
      const rem = p - z * nx * ny;
      const y = (rem / nx) | 0;
      const x = rem - y * nx;

      for (let d = 0; d < 6; d++) {
        const dx = d === 0 ? -1 : d === 1 ? 1 : 0;
        const dy = d === 2 ? -1 : d === 3 ? 1 : 0;
        const dz = d === 4 ? -1 : d === 5 ? 1 : 0;
        const a = x + dx, b = y + dy, c = z + dz;
        if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
        const q = a + b * nx + c * nx * ny;
        if (mask[q] === 1 && labels[q] === -1) { labels[q] = label; stack[top++] = q; }
      }
    }
    if (size > bestSize) { bestSize = size; bestLabel = label; }
    label++;
  }

  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = labels[i] === bestLabel ? 1 : 0;
  return out;
}

function fillHoles(mask, dims) {
  const [nx, ny, nz] = dims;
  const n = nx * ny * nz;
  const outside = new Uint8Array(n);
  const stack = new Int32Array(n);
  let top = 0;

  const push = (x, y, z) => {
    const p = x + y * nx + z * nx * ny;
    if (mask[p] === 0 && outside[p] === 0) { outside[p] = 1; stack[top++] = p; }
  };
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) { push(0, y, z); push(nx - 1, y, z); }
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) { push(x, 0, z); push(x, ny - 1, z); }
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { push(x, y, 0); push(x, y, nz - 1); }

  while (top > 0) {
    const p = stack[--top];
    const z = (p / (nx * ny)) | 0;
    const rem = p - z * nx * ny;
    const y = (rem / nx) | 0;
    const x = rem - y * nx;
    for (let d = 0; d < 6; d++) {
      const dx = d === 0 ? -1 : d === 1 ? 1 : 0;
      const dy = d === 2 ? -1 : d === 3 ? 1 : 0;
      const dz = d === 4 ? -1 : d === 5 ? 1 : 0;
      const a = x + dx, b = y + dy, c = z + dz;
      if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
      const q = a + b * nx + c * nx * ny;
      if (mask[q] === 0 && outside[q] === 0) { outside[q] = 1; stack[top++] = q; }
    }
  }

  const filled = new Uint8Array(n);
  for (let i = 0; i < n; i++) filled[i] = outside[i] === 0 ? 1 : 0;
  return filled;
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const pos = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Robust median/IQR normalisation inside the mask, then a percentile clip. */
export function normaliseWithinMask(data, mask) {
  const inside = [];
  for (let i = 0; i < data.length; i++) if (mask[i]) inside.push(data[i]);
  inside.sort((a, b) => a - b);

  const median = percentile(inside, 50);
  const iqr = percentile(inside, 75) - percentile(inside, 25) || 1;

  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    out[i] = mask[i] ? (data[i] - median) / iqr : 0;
  }
  return clipToUnit(out, mask);
}

/** Clip to the [1, 99] percentile of masked voxels and rescale to [0, 1]. */
export function clipToUnit(data, mask) {
  const inside = [];
  for (let i = 0; i < data.length; i++) if (mask[i]) inside.push(data[i]);
  inside.sort((a, b) => a - b);

  const lo = percentile(inside, 1);
  const hi = percentile(inside, 99);
  const span = hi - lo || 1;

  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    if (!mask[i]) { out[i] = 0; continue; }
    out[i] = Math.min(1, Math.max(0, (data[i] - lo) / span));
  }
  return out;
}

/** Apply pooled training-cohort site statistics, then re-clip (steps 5-6). */
export function applySiteNormalisation(data, mask, stats) {
  const mean = stats?.mean ?? 0;
  const sd = stats?.sd || 1;
  const z = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) z[i] = mask[i] ? (data[i] - mean) / sd : 0;
  return clipToUnit(z, mask);
}

/** Bilinear resize of a 2D array to size x size. */
export function resize2D(src, w, h, size = 224) {
  const out = new Float32Array(size * size);
  for (let j = 0; j < size; j++) {
    const sy = Math.min(h - 1, Math.max(0, ((j + 0.5) * h) / size - 0.5));
    const y0 = Math.floor(sy), y1 = Math.min(h - 1, y0 + 1), fy = sy - y0;
    for (let i = 0; i < size; i++) {
      const sx = Math.min(w - 1, Math.max(0, ((i + 0.5) * w) / size - 0.5));
      const x0 = Math.floor(sx), x1 = Math.min(w - 1, x0 + 1), fx = sx - x0;
      const a = src[y0 * w + x0], b = src[y0 * w + x1];
      const c = src[y1 * w + x0], d = src[y1 * w + x1];
      out[j * size + i] = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
                        + c * (1 - fx) * fy + d * fx * fy;
    }
  }
  return out;
}

/**
 * Extract one mid-plane slice at a given depth fraction.
 * plane: 'axial' | 'coronal' | 'sagittal' on an RAS volume.
 */
export function extractSlice(data, dims, plane, fraction = 0.5) {
  const [nx, ny, nz] = dims;
  let w, h, get;

  if (plane === 'axial') {
    const z = Math.min(nz - 1, Math.round(fraction * (nz - 1)));
    w = nx; h = ny;
    get = (x, y) => data[x + y * nx + z * nx * ny];
  } else if (plane === 'coronal') {
    const y = Math.min(ny - 1, Math.round(fraction * (ny - 1)));
    w = nx; h = nz;
    get = (x, z) => data[x + y * nx + z * nx * ny];
  } else {
    const x = Math.min(nx - 1, Math.round(fraction * (nx - 1)));
    w = ny; h = nz;
    get = (y, z) => data[x + y * nx + z * nx * ny];
  }

  const raw = new Float32Array(w * h);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      // Flip the vertical axis so superior/anterior renders upward.
      raw[(h - 1 - j) * w + i] = get(i, j);
    }
  }
  return { data: raw, w, h };
}

/**
 * Full pipeline: NIfTI volume -> a (3, 3, 224, 224) Float32Array,
 * one 3-channel image per plane, ready for the ONNX session.
 */
export function preprocess(volume, dims, affine, manifest) {
  const ras = reorientToRAS(volume, dims, affine);
  const mask = brainMask(ras.data, ras.dims, manifest.mask_fraction ?? 0.15);
  const robust = normaliseWithinMask(ras.data, mask);
  const sited = applySiteNormalisation(robust, mask, manifest.site_stats?.pooled);

  const size = manifest.input_size ?? 224;
  const planes = ['axial', 'coronal', 'sagittal'];
  const tensor = new Float32Array(3 * 3 * size * size);
  const slices = {};

  const mean = manifest.channel_mean ?? [0, 0, 0];
  const std = manifest.channel_std ?? [1, 1, 1];

  planes.forEach((plane, p) => {
    const s = extractSlice(sited, ras.dims, plane, 0.5);
    const resized = resize2D(s.data, s.w, s.h, size);
    slices[plane] = resized;
    for (let c = 0; c < 3; c++) {
      const base = (p * 3 + c) * size * size;
      for (let i = 0; i < size * size; i++) {
        tensor[base + i] = (resized[i] - mean[c]) / std[c];
      }
    }
  });

  return { tensor, slices, ras, mask, size };
}
