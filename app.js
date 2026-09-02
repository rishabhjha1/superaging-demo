import { preprocess, extractSlice, resize2D, reorientToRAS, brainMask,
         normaliseWithinMask, applySiteNormalisation } from './lib/preprocess.js';
import { checkOrientation, affineFromHeader } from './lib/orientation.js';
import { createSession, predict } from './lib/model.js';
import { rolloutMap } from './lib/rollout.js';
import { regionScores, aggregateRegional, regionalHeatmap, gridRegionNames,
         sliceFractions, REGIONS } from './lib/regions.js';
import { drawSlice, drawOverlay, drawGridPanel } from './lib/render.js';

const el = (id) => document.getElementById(id);
const state = { manifest: null, session: null, backend: null };

/* ---------------------------------------------------------------- bootstrap */

async function loadManifest() {
  const res = await fetch('models/manifest.json');
  if (!res.ok) {
    throw new Error('models/manifest.json is missing. Run scripts/export_onnx.py and commit its output into models/.');
  }
  return res.json();
}

async function ensureSession() {
  if (state.session) return state.session;
  const { session, backend } = await createSession(
    `models/${state.manifest.model_file}`, setStatus,
  );
  state.session = session;
  state.backend = backend;
  el('backend').textContent = backend === 'webgpu'
    ? 'Running on WebGPU'
    : 'Running on WebAssembly, single-threaded';
  return session;
}

function setStatus(text) {
  el('status').textContent = text;
}

function fail(message) {
  el('error').textContent = message;
  el('error').hidden = false;
  el('progress').hidden = true;
}

function clearError() {
  el('error').hidden = true;
}

/* ------------------------------------------------------------------ NIfTI */

async function readNifti(file) {
  const buf = await file.arrayBuffer();
  let data = buf;
  if (nifti.isCompressed(data)) data = nifti.decompress(data);
  if (!nifti.isNIFTI(data)) {
    throw new Error(`${file.name} is not a NIfTI file. Upload a .nii or .nii.gz volume.`);
  }

  const header = nifti.readHeader(data);
  const image = nifti.readImage(header, data);
  const typed = typedArrayFor(header, image);

  const dims = [header.dims[1], header.dims[2], header.dims[3]];
  if (header.dims[0] > 3 && header.dims[4] > 1) {
    throw new Error('This file has a time dimension. Upload a single 3D T1-weighted volume.');
  }

  const volume = new Float32Array(typed.length);
  const scl = header.scl_slope || 1;
  const inter = header.scl_inter || 0;
  for (let i = 0; i < typed.length; i++) volume[i] = typed[i] * scl + inter;

  return { volume, dims, header, affine: affineFromHeader(header) };
}

function typedArrayFor(header, buffer) {
  switch (header.datatypeCode) {
    case nifti.NIFTI1.TYPE_UINT8: return new Uint8Array(buffer);
    case nifti.NIFTI1.TYPE_INT16: return new Int16Array(buffer);
    case nifti.NIFTI1.TYPE_INT32: return new Int32Array(buffer);
    case nifti.NIFTI1.TYPE_FLOAT32: return new Float32Array(buffer);
    case nifti.NIFTI1.TYPE_FLOAT64: return new Float64Array(buffer);
    case nifti.NIFTI1.TYPE_INT8: return new Int8Array(buffer);
    case nifti.NIFTI1.TYPE_UINT16: return new Uint16Array(buffer);
    case nifti.NIFTI1.TYPE_UINT32: return new Uint32Array(buffer);
    default: throw new Error(`Unsupported NIfTI datatype code ${header.datatypeCode}.`);
  }
}

/* ------------------------------------------------------------------- main */

async function run(file) {
  clearError();
  el('results').hidden = true;
  el('progress').hidden = false;
  setStatus('Reading the volume');

  try {
    if (!state.manifest) {
      throw new Error(
        'The model is not deployed yet. models/manifest.json did not load, so '
        + 'there is nothing to run this scan through. Run scripts/export_onnx.py '
        + 'and commit its output into models/.');
    }
    const { volume, dims, affine } = await readNifti(file);

    const orientation = checkOrientation(affine, dims);
    renderOrientation(orientation, file.name, dims);

    await ensureSession();

    setStatus('Preprocessing');
    const pre = preprocess(volume, dims, affine, state.manifest);

    const { probability, attention, attnDims } =
      await predict(state.session, pre.tensor, pre.size, state.manifest, setStatus);

    setStatus('Building the explainability panels');
    const [L, V, N] = [attnDims[0], attnDims[1], attnDims[2]];

    const maps = {};
    ['axial', 'coronal', 'sagittal'].forEach((plane, i) => {
      maps[plane] = rolloutMap(attention, L, V, N, i, pre.size);
    });

    renderPrediction(probability);
    renderPanels(pre, maps);

    if (orientation.canonical) {
      renderTable(sweepRegions(volume, dims, affine, maps.axial, probability, pre.size));
      el('table-blocked').hidden = true;
      el('region-table').hidden = false;
    } else {
      el('region-table').hidden = true;
      el('table-blocked').hidden = false;
    }

    el('progress').hidden = true;
    el('results').hidden = false;
    el('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    console.error(err);
    fail(err.message || String(err));
  }
}

/**
 * Consistency sweep. The paper repeats the regional analysis at +/- 5% depth
 * increments so a region's score is not an artefact of one arbitrary slice.
 *
 * The attention map itself is computed once at the mid-slice; the sweep varies
 * the anatomical slice the grid is applied to. Running the full model at five
 * depths would be five times the inference cost, which single-threaded WASM
 * cannot absorb. The parity report records this difference.
 */
function sweepRegions(volume, dims, affine, axialMap, probability, size) {
  const ras = reorientToRAS(volume, dims, affine);
  const mask = brainMask(ras.data, ras.dims, state.manifest.mask_fraction ?? 0.15);
  const robust = normaliseWithinMask(ras.data, mask);
  const sited = applySiteNormalisation(robust, mask, state.manifest.site_stats?.pooled);

  const samples = sliceFractions().map((f) => {
    const s = extractSlice(sited, ras.dims, 'axial', f);
    const resized = resize2D(s.data, s.w, s.h, size);

    // Weight the attention map by local anatomy at this depth, so the sweep
    // reflects the slice rather than repeating the mid-slice five times.
    const weighted = new Float32Array(size * size);
    for (let i = 0; i < weighted.length; i++) weighted[i] = axialMap[i] * resized[i];

    const importance = regionScores(weighted, size, undefined, 'peak');
    const superagerScore = {};
    REGIONS.forEach((r) => { superagerScore[r] = importance[r] * probability; });
    return { importance, superagerScore };
  });

  return aggregateRegional(samples);
}

/* --------------------------------------------------------------- rendering */

function renderOrientation(o, name, dims) {
  el('file-name').textContent = name;
  el('file-dims').textContent = `${dims.join(' x ')} voxels, axis codes ${o.codes}`;

  const warn = el('orientation-warning');
  if (o.canonical && o.ok) {
    warn.hidden = true;
    return;
  }
  warn.hidden = false;
  el('orientation-detail').textContent = o.canonical
    ? o.problems.join(' ')
    : `This volume reports axis codes ${o.codes} rather than RAS. The 6x6 grid assumes RAS, so lobe labels would be unreliable. Attention maps below are still valid; the regional table is withheld.`;
}

function renderPrediction(p) {
  el('probability').textContent = p.toFixed(3);
  el('verdict').textContent = p >= (state.manifest.threshold ?? 0.5)
    ? 'Classified as SuperAger'
    : 'Classified as typical ager';
  el('prob-bar').style.setProperty('--fill', `${(p * 100).toFixed(1)}%`);
  el('threshold-note').textContent =
    `Decision threshold ${(state.manifest.threshold ?? 0.5).toFixed(2)}, chosen on the validation pool.`;
}

function renderPanels(pre, maps) {
  drawSlice(el('panel-axial'), pre.slices.axial, pre.size);
  ['axial', 'coronal', 'sagittal'].forEach((plane) => {
    drawOverlay(el(`panel-${plane}-attn`), pre.slices[plane], maps[plane], pre.size);
  });

  const scores = regionScores(maps.axial, pre.size, undefined, 'peak');
  drawGridPanel(el('panel-grid'), regionalHeatmap(scores), gridRegionNames(), pre.size);
}

function renderTable(rows) {
  const body = el('region-body');
  body.innerHTML = '';
  rows.forEach((r) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${r.region}</td>
      <td>${r.I_R.toFixed(3)} <span class="sd">± ${r.I_R_sd.toFixed(3)}</span></td>
      <td>${r.S_R.toFixed(3)} <span class="sd">± ${r.S_R_sd.toFixed(3)}</span></td>`;
    body.appendChild(tr);
  });
}

/* ----------------------------------------------------------------- wiring */

function wire() {
  const drop = el('drop');
  const input = el('file');

  drop.addEventListener('click', () => input.click());
  drop.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => {
    if (input.files.length) run(input.files[0]);
  });

  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add('is-over');
  }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
  }));
  drop.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length) run(e.dataTransfer.files[0]);
  });

  document.querySelectorAll('[data-sample]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      clearError();
      setStatus('Fetching the sample scan');
      el('progress').hidden = false;
      try {
        const url = btn.dataset.sample;
        const res = await fetch(url);
        if (!res.ok) throw new Error('The sample scan is not in this deployment. See samples/README.md.');
        const blob = await res.blob();
        run(new File([blob], url.split('/').pop()));
      } catch (err) {
        fail(err.message);
      }
    });
  });
}

// Listeners are attached before the manifest is fetched, and unconditionally.
// If they were attached inside the .then() below, a missing manifest would
// leave the drop zone and the sample button inert with no explanation: the page
// would render correctly and do nothing at all when clicked.
wire();

loadManifest()
  .then((m) => {
    state.manifest = m;
    el('model-meta').textContent =
      `${m.model_file}, exported ${m.exported_at}. Parity against PyTorch: max Δp = ${m.parity_max_delta}.`;
  })
  .catch((err) => {
    fail(`${err.message} The page is loaded and will accept a file, but cannot run one until the model is deployed.`);
    el('model-meta').textContent = 'No model deployed.';
  });
