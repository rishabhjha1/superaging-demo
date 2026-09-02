/** Canvas rendering for the explainability panels. */

/** Grayscale MRI slice. */
export function drawSlice(canvas, slice, size) {
  const ctx = canvas.getContext('2d');
  canvas.width = size;
  canvas.height = size;
  const img = ctx.createImageData(size, size);

  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < slice.length; i++) {
    if (slice[i] < lo) lo = slice[i];
    if (slice[i] > hi) hi = slice[i];
  }
  const span = hi - lo || 1;

  for (let i = 0; i < size * size; i++) {
    const v = Math.round(((slice[i] - lo) / span) * 255);
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * Saliency overlaid on the slice. `alpha` scales with the saliency value so
 * low-attention areas stay readable as anatomy rather than being painted over.
 */
export function drawOverlay(canvas, slice, saliency, size, opacity = 0.62) {
  const ctx = canvas.getContext('2d');
  canvas.width = size;
  canvas.height = size;
  const img = ctx.createImageData(size, size);

  let slo = Infinity, shi = -Infinity;
  for (let i = 0; i < slice.length; i++) {
    if (slice[i] < slo) slo = slice[i];
    if (slice[i] > shi) shi = slice[i];
  }
  const sspan = shi - slo || 1;

  let alo = Infinity, ahi = -Infinity;
  for (let i = 0; i < saliency.length; i++) {
    if (saliency[i] < alo) alo = saliency[i];
    if (saliency[i] > ahi) ahi = saliency[i];
  }
  const aspan = ahi - alo || 1;

  for (let i = 0; i < size * size; i++) {
    const g = ((slice[i] - slo) / sspan) * 255;
    const a = (saliency[i] - alo) / aspan;
    const [r, gg, b] = inferno(a);
    const w = opacity * a;
    img.data[i * 4] = Math.round(g * (1 - w) + r * w);
    img.data[i * 4 + 1] = Math.round(g * (1 - w) + gg * w);
    img.data[i * 4 + 2] = Math.round(g * (1 - w) + b * w);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** 6x6 regional heatmap with region initials, matching Fig. 2's last panel. */
export function drawGridPanel(canvas, heat, names, size = 224) {
  const ctx = canvas.getContext('2d');
  canvas.width = size;
  canvas.height = size;
  const g = heat.length;
  const cell = size / g;

  let lo = Infinity, hi = -Infinity;
  heat.forEach((row) => row.forEach((v) => {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }));
  const span = hi - lo || 1;

  ctx.font = `${Math.round(cell * 0.32)}px ui-sans-serif, system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < g; i++) {
    for (let j = 0; j < g; j++) {
      const a = (heat[i][j] - lo) / span;
      const [r, gg, b] = inferno(a);
      ctx.fillStyle = `rgb(${r}, ${gg}, ${b})`;
      ctx.fillRect(j * cell, i * cell, cell, cell);
      ctx.fillStyle = a > 0.55 ? 'rgba(20,20,26,0.85)' : 'rgba(245,245,250,0.85)';
      ctx.fillText(names[i][j][0], j * cell + cell / 2, i * cell + cell / 2);
    }
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.16)';
  ctx.lineWidth = 1;
  for (let k = 1; k < g; k++) {
    ctx.beginPath();
    ctx.moveTo(k * cell, 0); ctx.lineTo(k * cell, size);
    ctx.moveTo(0, k * cell); ctx.lineTo(size, k * cell);
    ctx.stroke();
  }
}

/** Perceptually ordered colormap; approximates matplotlib's inferno. */
function inferno(t) {
  const stops = [
    [0.0, 0, 0, 4], [0.15, 40, 11, 84], [0.3, 101, 21, 110],
    [0.45, 159, 42, 99], [0.6, 212, 72, 66], [0.75, 245, 125, 21],
    [0.9, 250, 193, 39], [1.0, 252, 255, 164],
  ];
  const x = Math.min(1, Math.max(0, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, r0, g0, b0] = stops[i];
    const [p1, r1, g1, b1] = stops[i + 1];
    if (x >= p0 && x <= p1) {
      const f = (x - p0) / (p1 - p0 || 1);
      return [
        Math.round(r0 + (r1 - r0) * f),
        Math.round(g0 + (g1 - g0) * f),
        Math.round(b0 + (b1 - b0) * f),
      ];
    }
  }
  return [252, 255, 164];
}
