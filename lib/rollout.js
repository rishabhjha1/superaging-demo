/**
 * Attention rollout (paper Sec. 3.5).
 *
 * Recursively multiplies residual-augmented, row-normalised attention matrices
 * across encoder layers, then reads the CLS row and upsamples the patch-level
 * attention to slice resolution.
 *
 * Per-plane maps are kept separate, as the paper reports them, because the
 * three planes are mean-fused only at the embedding stage.
 */

import { resize2D } from './preprocess.js';

/**
 * @param {Float32Array} attn  flat (L, V, N, N) head-averaged attention
 * @param {number} L           encoder layers
 * @param {number} V           views (3)
 * @param {number} N           tokens (197)
 * @param {number} view        which plane to roll out
 * @returns {Float32Array}     N x N rollout matrix
 */
export function rolloutMatrix(attn, L, V, N, view) {
  let acc = identity(N);

  for (let l = 0; l < L; l++) {
    const layer = new Float32Array(N * N);
    const base = (l * V + view) * N * N;

    // Residual augmentation: A_hat = 0.5 * A + 0.5 * I, then renormalise rows.
    for (let r = 0; r < N; r++) {
      let sum = 0;
      for (let c = 0; c < N; c++) {
        const v = 0.5 * attn[base + r * N + c] + (r === c ? 0.5 : 0);
        layer[r * N + c] = v;
        sum += v;
      }
      if (sum > 0) for (let c = 0; c < N; c++) layer[r * N + c] /= sum;
    }
    acc = matmul(layer, acc, N);
  }
  return acc;
}

function identity(N) {
  const m = new Float32Array(N * N);
  for (let i = 0; i < N; i++) m[i * N + i] = 1;
  return m;
}

function matmul(a, b, N) {
  const out = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < N; k++) {
      const av = a[i * N + k];
      if (av === 0) continue;
      for (let j = 0; j < N; j++) out[i * N + j] += av * b[k * N + j];
    }
  }
  return out;
}

/**
 * CLS-row patch attention, reshaped to the patch grid and upsampled.
 * @returns {Float32Array} size x size map normalised to [0, 1]
 */
export function rolloutMap(attn, L, V, N, view, size = 224) {
  const acc = rolloutMatrix(attn, L, V, N, view);
  const nPatches = N - 1;
  const grid = Math.round(Math.sqrt(nPatches));

  const patch = new Float32Array(nPatches);
  for (let i = 0; i < nPatches; i++) patch[i] = acc[0 * N + (i + 1)];

  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < nPatches; i++) {
    if (patch[i] < lo) lo = patch[i];
    if (patch[i] > hi) hi = patch[i];
  }
  const span = hi - lo || 1;
  for (let i = 0; i < nPatches; i++) patch[i] = (patch[i] - lo) / span;

  return resize2D(patch, grid, grid, size);
}
