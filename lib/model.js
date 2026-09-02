/**
 * ONNX Runtime Web session wrapper.
 *
 * The exported graph takes a (3, 3, 224, 224) tensor (one 3-channel image per
 * plane) and returns:
 *   logits     (1, 2)
 *   attention  (L, 3, 197, 197)  head-averaged, per layer, per plane
 *
 * Test-time augmentation averages four views: original, horizontal flip, and
 * rescalings x1.05 and x0.95, matching the paper's Sec. 3.4. Attention maps are
 * taken from the unaugmented pass only, so the explainability panels correspond
 * to the slice actually shown on screen.
 */

const CDN = 'https://cdnjs.cloudflare.com/ajax/libs/onnxruntime-web/1.19.2/';

export async function createSession(modelUrl, onProgress = () => {}) {
  const ort = window.ort;
  ort.env.wasm.wasmPaths = CDN;
  ort.env.wasm.numThreads = 1; // GitHub Pages cannot set the COOP/COEP headers
                               // that SharedArrayBuffer needs, so threads are
                               // unavailable. See README, "Why it is slow".

  onProgress('Downloading the model');
  const buffer = await fetchWithProgress(modelUrl, onProgress);

  const providers = [];
  if ('gpu' in navigator) providers.push('webgpu');
  providers.push('wasm');

  onProgress('Starting the inference session');
  let session;
  let backend;
  for (const ep of providers) {
    try {
      session = await ort.InferenceSession.create(buffer, { executionProviders: [ep] });
      backend = ep;
      break;
    } catch (err) {
      console.warn(`Execution provider ${ep} unavailable:`, err);
    }
  }
  if (!session) throw new Error('No execution provider could start. Try a Chromium-based browser.');

  return { session, backend };
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load the model (HTTP ${res.status}). Check that models/ contains the exported file.`);

  const total = Number(res.headers.get('content-length')) || 0;
  if (!total || !res.body) return res.arrayBuffer();

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(`Downloading the model, ${Math.round((received / total) * 100)}%`);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  chunks.forEach((c) => { out.set(c, offset); offset += c.length; });
  return out.buffer;
}

function hflip(tensor, size) {
  const out = new Float32Array(tensor.length);
  const planes = tensor.length / (size * size);
  for (let p = 0; p < planes; p++) {
    const base = p * size * size;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        out[base + y * size + x] = tensor[base + y * size + (size - 1 - x)];
      }
    }
  }
  return out;
}

function rescale(tensor, size, factor) {
  const out = new Float32Array(tensor.length);
  const planes = tensor.length / (size * size);
  const c = (size - 1) / 2;
  for (let p = 0; p < planes; p++) {
    const base = p * size * size;
    for (let y = 0; y < size; y++) {
      const sy = Math.min(size - 1, Math.max(0, c + (y - c) / factor));
      const y0 = Math.floor(sy), y1 = Math.min(size - 1, y0 + 1), fy = sy - y0;
      for (let x = 0; x < size; x++) {
        const sx = Math.min(size - 1, Math.max(0, c + (x - c) / factor));
        const x0 = Math.floor(sx), x1 = Math.min(size - 1, x0 + 1), fx = sx - x0;
        const a = tensor[base + y0 * size + x0], b = tensor[base + y0 * size + x1];
        const d = tensor[base + y1 * size + x0], e = tensor[base + y1 * size + x1];
        out[base + y * size + x] = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
                                 + d * (1 - fx) * fy + e * fx * fy;
      }
    }
  }
  return out;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

/**
 * Run the model with TTA.
 * @returns {{probability: number, attention: Float32Array, attnDims: number[]}}
 */
export async function predict(session, tensor, size, manifest, onProgress = () => {}) {
  const ort = window.ort;
  const inputName = session.inputNames[0];
  const shape = [3, 3, size, size];

  const views = manifest.tta === false
    ? [tensor]
    : [tensor, hflip(tensor, size), rescale(tensor, size, 1.05), rescale(tensor, size, 0.95)];

  let attention = null;
  let attnDims = null;
  const probs = [];

  for (let i = 0; i < views.length; i++) {
    onProgress(`Running the model, pass ${i + 1} of ${views.length}`);
    const feeds = { [inputName]: new ort.Tensor('float32', views[i], shape) };
    // eslint-disable-next-line no-await-in-loop
    const out = await session.run(feeds);

    const logits = Array.from(out.logits.data);
    probs.push(softmax(logits)[manifest.superager_index ?? 1]);

    if (i === 0 && out.attention) {
      attention = out.attention.data;
      attnDims = out.attention.dims;
    }
    // Yield to the event loop so the progress text repaints.
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 0));
  }

  return {
    probability: probs.reduce((a, b) => a + b, 0) / probs.length,
    attention,
    attnDims,
  };
}
