/**
 * Minimal NIfTI-1 / NIfTI-2 reader.
 *
 * This replaces nifti-reader-js. That library is not loadable as a plain
 * script tag: 0.6.x ships CommonJS and 0.7+ ships an ES module, so neither
 * creates the `nifti` global the page used to assume, and the cdnjs path the
 * page pointed at does not exist at all.
 *
 * The header is a fixed 348-byte struct (540 for NIfTI-2) and gzip is built
 * into the browser as DecompressionStream, so the whole dependency is about
 * two hundred lines. Vendoring it also means the demo has no CDN in its
 * critical path, which matters for a page reviewers have to be able to open.
 *
 * The exported surface is exactly what app.js calls:
 *   isCompressed, decompress, isNIFTI, readHeader, readImage, NIFTI1.TYPE_*
 * plus hasExtension for parity with the old API.
 *
 * NOT a general-purpose reader. It handles scalar 3D volumes in the datatypes
 * app.js switches on. It does not do extensions, complex or RGB types, or
 * NIfTI pairs (.hdr/.img).
 */

const NIFTI1_HEADER_SIZE = 348;
const NIFTI2_HEADER_SIZE = 540;

export const NIFTI1 = {
  TYPE_UINT8: 2,
  TYPE_INT16: 4,
  TYPE_INT32: 8,
  TYPE_FLOAT32: 16,
  TYPE_FLOAT64: 64,
  TYPE_INT8: 256,
  TYPE_UINT16: 512,
  TYPE_UINT32: 768,
  TYPE_INT64: 1024,
  TYPE_UINT64: 1280,
};

export const NIFTI2 = { ...NIFTI1 };

const BYTES_PER_VOXEL = {
  2: 1, 4: 2, 8: 4, 16: 4, 64: 8, 256: 1, 512: 2, 768: 4, 1024: 8, 1280: 8,
};

/* ------------------------------------------------------------ compression */

export function isCompressed(buffer) {
  if (!buffer || buffer.byteLength < 2) return false;
  const v = new DataView(buffer);
  return v.getUint8(0) === 0x1f && v.getUint8(1) === 0x8b;
}

/**
 * Gunzip via the browser's own DecompressionStream.
 *
 * Async, unlike nifti-reader-js's synchronous decompress, so the caller must
 * await it. Awaiting a non-promise is harmless, so `await decompress(...)` is
 * correct either way.
 */
export async function decompress(buffer) {
  if (typeof DecompressionStream !== 'function') {
    throw new Error(
      'This browser cannot gunzip .nii.gz files (no DecompressionStream). '
      + 'Uncompress the volume to .nii and try again.');
  }
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

/* ---------------------------------------------------------------- sniffing */

function headerFlavour(buffer) {
  if (!buffer || buffer.byteLength < NIFTI1_HEADER_SIZE) return null;
  const v = new DataView(buffer);

  // sizeof_hdr is the first field and doubles as the endianness probe.
  if (v.getInt32(0, true) === NIFTI1_HEADER_SIZE) return { version: 1, little: true };
  if (v.getInt32(0, false) === NIFTI1_HEADER_SIZE) return { version: 1, little: false };

  if (buffer.byteLength >= NIFTI2_HEADER_SIZE) {
    if (v.getInt32(0, true) === NIFTI2_HEADER_SIZE) return { version: 2, little: true };
    if (v.getInt32(0, false) === NIFTI2_HEADER_SIZE) return { version: 2, little: false };
  }
  return null;
}

export function isNIFTI1(buffer) {
  const f = headerFlavour(buffer);
  return !!f && f.version === 1;
}

export function isNIFTI2(buffer) {
  const f = headerFlavour(buffer);
  return !!f && f.version === 2;
}

export function isNIFTI(buffer) {
  return headerFlavour(buffer) !== null;
}

/* ----------------------------------------------------------------- header */

function readString(view, offset, length) {
  let s = '';
  for (let i = 0; i < length; i++) {
    const c = view.getUint8(offset + i);
    if (c === 0) break;
    s += String.fromCharCode(c);
  }
  return s;
}

export function readHeader(buffer) {
  const flavour = headerFlavour(buffer);
  if (!flavour) throw new Error('That file does not appear to be NIfTI.');
  return flavour.version === 1
    ? readHeader1(buffer, flavour.little)
    : readHeader2(buffer, flavour.little);
}

function readHeader1(buffer, little) {
  const v = new DataView(buffer);
  const dims = [];
  for (let i = 0; i < 8; i++) dims.push(v.getInt16(40 + i * 2, little));

  const pixDims = [];
  for (let i = 0; i < 8; i++) pixDims.push(v.getFloat32(76 + i * 4, little));

  const srow = [
    [v.getFloat32(280, little), v.getFloat32(284, little), v.getFloat32(288, little), v.getFloat32(292, little)],
    [v.getFloat32(296, little), v.getFloat32(300, little), v.getFloat32(304, little), v.getFloat32(308, little)],
    [v.getFloat32(312, little), v.getFloat32(316, little), v.getFloat32(320, little), v.getFloat32(324, little)],
  ];

  return {
    version: 1,
    littleEndian: little,
    dims,
    pixDims,
    datatypeCode: v.getInt16(70, little),
    numBitsPerVoxel: v.getInt16(72, little),
    scl_slope: v.getFloat32(112, little),
    scl_inter: v.getFloat32(116, little),
    cal_max: v.getFloat32(124, little),
    cal_min: v.getFloat32(128, little),
    vox_offset: v.getFloat32(108, little),
    qform_code: v.getInt16(252, little),
    sform_code: v.getInt16(254, little),
    quatern_b: v.getFloat32(256, little),
    quatern_c: v.getFloat32(260, little),
    quatern_d: v.getFloat32(264, little),
    qoffset_x: v.getFloat32(268, little),
    qoffset_y: v.getFloat32(272, little),
    qoffset_z: v.getFloat32(276, little),
    srow_x: srow[0],
    srow_y: srow[1],
    srow_z: srow[2],
    affine: buildAffine(srow, {
      qform_code: v.getInt16(252, little),
      sform_code: v.getInt16(254, little),
      quatern_b: v.getFloat32(256, little),
      quatern_c: v.getFloat32(260, little),
      quatern_d: v.getFloat32(264, little),
      qoffset_x: v.getFloat32(268, little),
      qoffset_y: v.getFloat32(272, little),
      qoffset_z: v.getFloat32(276, little),
      pixDims,
    }),
    description: readString(v, 148, 80),
    extensionFlag: [buffer.byteLength > 348 ? v.getUint8(348) : 0, 0, 0, 0],
  };
}

function readHeader2(buffer, little) {
  const v = new DataView(buffer);
  const big = (o) => Number(v.getBigInt64(o, little));

  const dims = [];
  for (let i = 0; i < 8; i++) dims.push(big(16 + i * 8));

  const pixDims = [];
  for (let i = 0; i < 8; i++) pixDims.push(v.getFloat64(104 + i * 8, little));

  const srow = [
    [v.getFloat64(400, little), v.getFloat64(408, little), v.getFloat64(416, little), v.getFloat64(424, little)],
    [v.getFloat64(432, little), v.getFloat64(440, little), v.getFloat64(448, little), v.getFloat64(456, little)],
    [v.getFloat64(464, little), v.getFloat64(472, little), v.getFloat64(480, little), v.getFloat64(488, little)],
  ];

  const meta = {
    qform_code: v.getInt32(344, little),
    sform_code: v.getInt32(348, little),
    quatern_b: v.getFloat64(352, little),
    quatern_c: v.getFloat64(360, little),
    quatern_d: v.getFloat64(368, little),
    qoffset_x: v.getFloat64(376, little),
    qoffset_y: v.getFloat64(384, little),
    qoffset_z: v.getFloat64(392, little),
    pixDims,
  };

  return {
    version: 2,
    littleEndian: little,
    dims,
    pixDims,
    datatypeCode: v.getInt16(12, little),
    numBitsPerVoxel: v.getInt16(14, little),
    scl_slope: v.getFloat64(176, little),
    scl_inter: v.getFloat64(184, little),
    cal_max: v.getFloat64(192, little),
    cal_min: v.getFloat64(200, little),
    vox_offset: big(168),
    ...meta,
    srow_x: srow[0],
    srow_y: srow[1],
    srow_z: srow[2],
    affine: buildAffine(srow, meta),
    description: readString(v, 240, 80),
    extensionFlag: [buffer.byteLength > 540 ? v.getUint8(540) : 0, 0, 0, 0],
  };
}

/**
 * Voxel-to-world affine, sform first then qform, matching the NIfTI-1 spec's
 * precedence. Falls back to a diagonal scaling from pixdim when neither code
 * is set, which is what most tools do for headers that predate the codes.
 */
function buildAffine(srow, h) {
  if (h.sform_code > 0) return [srow[0], srow[1], srow[2], [0, 0, 0, 1]];

  if (h.qform_code > 0) {
    const b = h.quatern_b, c = h.quatern_c, d = h.quatern_d;
    const a = Math.sqrt(Math.max(0, 1 - (b * b + c * c + d * d)));
    const [dx, dy, dz] = [h.pixDims[1], h.pixDims[2], h.pixDims[3]];
    const qfac = h.pixDims[0] < 0 ? -1 : 1;
    const R = [
      [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
      [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
      [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ];
    return [
      [R[0][0] * dx, R[0][1] * dy, R[0][2] * dz * qfac, h.qoffset_x],
      [R[1][0] * dx, R[1][1] * dy, R[1][2] * dz * qfac, h.qoffset_y],
      [R[2][0] * dx, R[2][1] * dy, R[2][2] * dz * qfac, h.qoffset_z],
      [0, 0, 0, 1],
    ];
  }

  return [
    [h.pixDims[1] || 1, 0, 0, 0],
    [0, h.pixDims[2] || 1, 0, 0],
    [0, 0, h.pixDims[3] || 1, 0],
    [0, 0, 0, 1],
  ];
}

export function hasExtension(header) {
  return header.extensionFlag[0] !== 0;
}

/* ------------------------------------------------------------------ image */

export function readImage(header, buffer) {
  const bytes = BYTES_PER_VOXEL[header.datatypeCode];
  if (!bytes) {
    throw new Error(`Unsupported NIfTI datatype code ${header.datatypeCode}.`);
  }

  let n = 1;
  for (let i = 1; i <= Math.min(header.dims[0], 7); i++) {
    n *= Math.max(1, header.dims[i]);
  }

  const offset = Math.max(header.vox_offset | 0, header.version === 1
    ? NIFTI1_HEADER_SIZE : NIFTI2_HEADER_SIZE);
  const length = n * bytes;

  if (offset + length > buffer.byteLength) {
    throw new Error(
      `The file is truncated: the header describes ${length} bytes of image `
      + `data at offset ${offset}, but the file is only ${buffer.byteLength} bytes.`);
  }

  const slice = buffer.slice(offset, offset + length);

  // Byte-swap in place for big-endian files; every typed-array view below
  // assumes host order, which is little-endian everywhere this runs.
  if (!header.littleEndian && bytes > 1) swapInPlace(slice, bytes);
  return slice;
}

function swapInPlace(buffer, width) {
  const b = new Uint8Array(buffer);
  for (let i = 0; i < b.length; i += width) {
    for (let j = 0; j < width >> 1; j++) {
      const t = b[i + j];
      b[i + j] = b[i + width - 1 - j];
      b[i + width - 1 - j] = t;
    }
  }
}

/* --------------------------------------------------------------- default */

const api = {
  NIFTI1, NIFTI2,
  isCompressed, decompress,
  isNIFTI, isNIFTI1, isNIFTI2,
  readHeader, readImage, hasExtension,
};

export default api;
