/**
 * Grid-based regional analysis (paper Sec. 3.5, Table 2).
 * Line-for-line port of regions.py, including the peak-scaling convention.
 *
 * I_R      mean attention weight over the region's cells
 * S_R      I_R * P(SuperAger)
 * sigma_R  SD of I_R across the slice-offset sweep
 *
 * CAVEAT 1. The grid is a coarse proxy, not an atlas. A 6x6 partition of a
 * mid-axial slice approximates lobar location; cell-to-region assignment near
 * boundaries is approximate.
 *
 * CAVEAT 2. I_R is reported relative to peak. With scaling 'peak' the five
 * region means are divided by the largest within each slice, so the top region
 * sits near 1.0. This is a display convention; it does not change the model,
 * the ranking, or any comparison.
 */

export const REGIONS = ['Frontal', 'Parietal', 'Temporal', 'Central', 'Subcortical'];

const CODE_TO_REGION = {
  F: 'Frontal', P: 'Parietal', T: 'Temporal', C: 'Central', S: 'Subcortical',
};

// Default 6x6 layout for a canonical (RAS) axial mid-slice.
// Rows run posterior -> anterior along axis 0; columns left -> right on axis 1.
export const DEFAULT_GRID = [
  'FFFFFF',
  'FFFFFF',
  'TCCCCT',
  'TSSSST',
  'PPPPPP',
  'PPPPPP',
].map((row) => row.split(''));

export function gridRegionNames(grid = DEFAULT_GRID) {
  return grid.map((row) => row.map((code) => {
    const name = CODE_TO_REGION[code];
    if (!name) throw new Error(`Unknown region code '${code}'`);
    return name;
  }));
}

/**
 * Adaptive average pool onto a gridSize x gridSize grid, matching
 * torch.nn.functional.adaptive_avg_pool2d's floor/ceil bin boundaries.
 */
export function poolToGrid(saliency, size, gridSize = 6) {
  const out = new Float32Array(gridSize * gridSize);
  for (let gy = 0; gy < gridSize; gy++) {
    const y0 = Math.floor((gy * size) / gridSize);
    const y1 = Math.ceil(((gy + 1) * size) / gridSize);
    for (let gx = 0; gx < gridSize; gx++) {
      const x0 = Math.floor((gx * size) / gridSize);
      const x1 = Math.ceil(((gx + 1) * size) / gridSize);
      let sum = 0, count = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) { sum += saliency[y * size + x]; count++; }
      }
      out[gy * gridSize + gx] = count ? sum / count : 0;
    }
  }
  return out;
}

/** Pool a saliency map into per-region means. scaling: 'peak' | 'raw' | 'sum'. */
export function regionScores(saliency, size, grid = DEFAULT_GRID, scaling = 'peak') {
  const names = gridRegionNames(grid);
  const g = names.length;
  const pooled = poolToGrid(saliency, size, g);

  const buckets = {};
  REGIONS.forEach((r) => { buckets[r] = []; });
  for (let i = 0; i < g; i++) {
    for (let j = 0; j < g; j++) buckets[names[i][j]].push(pooled[i * g + j]);
  }

  const values = {};
  REGIONS.forEach((r) => {
    const v = buckets[r];
    values[r] = v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
  });

  if (scaling === 'raw') return values;
  const finite = Object.values(values).filter((v) => Number.isFinite(v));
  if (scaling === 'sum') {
    const total = finite.reduce((a, b) => a + b, 0) + 1e-8;
    return mapValues(values, (v) => v / total);
  }
  if (scaling === 'peak') {
    const peak = Math.max(...finite) + 1e-8;
    return mapValues(values, (v) => v / peak);
  }
  throw new Error(`Unknown scaling '${scaling}'; use peak | raw | sum`);
}

function mapValues(obj, fn) {
  const out = {};
  Object.keys(obj).forEach((k) => { out[k] = fn(obj[k]); });
  return out;
}

/** Depth fractions for the consistency sweep, e.g. 0.40 ... 0.60. */
export function sliceFractions(centre = 0.5, increment = 0.05, steps = 2) {
  const out = [];
  for (let k = -steps; k <= steps; k++) out.push(Number((centre + k * increment).toFixed(4)));
  return out;
}

/**
 * Aggregate per-slice scores into the paper's Table 2 shape.
 * @param {Array<{importance: Object, superagerScore: Object}>} samples
 * @returns {Array} rows sorted by descending I_R, with a rank column
 */
export function aggregateRegional(samples) {
  if (!samples.length) return [];

  const rows = REGIONS.map((region) => {
    const imp = samples.map((s) => s.importance[region]);
    const sa = samples.map((s) => s.superagerScore[region]);
    return {
      region,
      I_R: mean(imp),
      I_R_sd: sd(imp),
      S_R: mean(sa),
      S_R_sd: sd(sa),
      n: imp.length,
    };
  });

  rows.sort((a, b) => b.I_R - a.I_R);
  rows.forEach((r, i) => { r.rank = i + 1; });
  return rows;
}

function mean(a) {
  const f = a.filter(Number.isFinite);
  return f.length ? f.reduce((x, y) => x + y, 0) / f.length : NaN;
}

function sd(a) {
  const f = a.filter(Number.isFinite);
  if (f.length < 2) return 0;
  const m = mean(f);
  // ddof = 1, matching numpy's np.nanstd(..., ddof=1) in regions.py
  return Math.sqrt(f.reduce((s, v) => s + (v - m) ** 2, 0) / (f.length - 1));
}

/** Paint per-region scores back onto the grid, for the figure's last panel. */
export function regionalHeatmap(scores, grid = DEFAULT_GRID) {
  const names = gridRegionNames(grid);
  return names.map((row) => row.map((r) => scores[r] ?? 0));
}
