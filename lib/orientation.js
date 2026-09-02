/**
 * Orientation guard.
 *
 * DEFAULT_GRID in regions.js assumes a canonical RAS axial slice. A flipped or
 * transposed volume swaps frontal and parietal labels while leaving every
 * number in Table 2 unchanged and entirely plausible-looking. That is the one
 * failure mode a reviewer must never hit silently, so this runs on every upload
 * and the regional table is withheld when it fails.
 *
 * This is the browser equivalent of scripts/check_grid_orientation.py in the
 * main repository.
 */

import { affineToAxisCodes } from './preprocess.js';

export function checkOrientation(affine, dims) {
  const problems = [];
  const codes = affineToAxisCodes(affine);

  const det = determinant3(affine);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-6) {
    problems.push('The affine is singular or missing, so anatomical direction cannot be established.');
  }

  const unique = new Set(codes.map((c) => ({ R: 'x', L: 'x', A: 'y', P: 'y', S: 'z', I: 'z' }[c])));
  if (unique.size !== 3) {
    problems.push(`Axis codes ${codes.join('')} do not cover all three anatomical axes.`);
  }

  const voxels = dims[0] * dims[1] * dims[2];
  if (voxels < 100000) {
    problems.push(`Volume is only ${dims.join(' x ')}; this is smaller than any usable T1w scan.`);
  }

  const aspect = Math.max(...dims) / Math.min(...dims);
  if (aspect > 6) {
    problems.push(`Volume is very anisotropic (${dims.join(' x ')}); mid-slice extraction may not be anatomically meaningful.`);
  }

  return {
    ok: problems.length === 0,
    codes: codes.join(''),
    canonical: codes.join('') === 'RAS',
    problems,
  };
}

function determinant3(m) {
  if (!m) return NaN;
  const [a, b, c] = [m[0][0], m[0][1], m[0][2]];
  const [d, e, f] = [m[1][0], m[1][1], m[1][2]];
  const [g, h, i] = [m[2][0], m[2][1], m[2][2]];
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/** Read a 4x4 affine out of a nifti-reader-js header. */
export function affineFromHeader(header) {
  if (header.affine && header.affine.length === 4) return header.affine;
  if (header.sform_code > 0) {
    return [
      header.srow_x, header.srow_y, header.srow_z, [0, 0, 0, 1],
    ];
  }
  return null;
}
