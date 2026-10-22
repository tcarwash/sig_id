import assert from "node:assert/strict";
import test from "node:test";

import { createDemodExport, downloadDemodExport } from "../sigid/web/demod-export.mjs";

test("demod export retains capture provenance, exact settings, decisions and protocol uncertainty", () => {
  const result = {
    demodulator: "noncoherent-mfsk-tone-correlator",
    symbols: [0, 3, 2],
    bits: "001110",
    soft_confidence: [0.9, 0.7, 0.8],
    symbol_sample_bounds: [[1200, 1440], [1440, 1680], [1680, 1920]],
    protocol: null,
  };
  const settings = { sample_start: 1200, sample_count: 720, symbol_rate: 50, tone_count: 4 };
  const artifact = createDemodExport({
    capture: { id: "capture-1", sha256: "a".repeat(64), sample_rate: 12000, sample_count: 9600 },
    mode: "mfsk",
    result,
    settings,
    createdAt: "2026-10-21T00:00:00.000Z",
  });
  assert.equal(artifact.artifact_type, "protocol-independent-demodulation");
  assert.equal(artifact.source_capture.capture_id, "capture-1");
  assert.equal(artifact.source_capture.sha256, "a".repeat(64));
  assert.deepEqual(artifact.settings, settings);
  assert.deepEqual(artifact.result.symbols, [0, 3, 2]);
  assert.equal(artifact.protocol, null);
  assert.deepEqual(artifact.bit_annotations, []);
  assert.equal(artifact.created_at, "2026-10-21T00:00:00.000Z");
});

test("demod export validates required source and result fields", () => {
  assert.throws(() => createDemodExport({}), TypeError);
  assert.throws(() => createDemodExport({ capture: { id: "x", sha256: "x" }, mode: "", result: {}, settings: {} }), TypeError);
});

test("download action creates a JSON file link and releases its object URL", () => {
  let clicked = false;
  let revoked = null;
  let removed = false;
  let release = null;
  const link = { click() { clicked = true; }, remove() { removed = true; } };
  const artifact = { protocol: null, result: { symbols: [1, 0] } };
  downloadDemodExport(artifact, "capture-mfsk.json", {
    documentRef: {
      body: { append: (element) => assert.equal(element, link) },
      createElement: (tag) => { assert.equal(tag, "a"); return link; },
    },
    urlApi: {
      createObjectURL: (blob) => { assert.ok(blob instanceof Blob); return "blob:demod-export"; },
      revokeObjectURL: (url) => { revoked = url; },
    },
    timerApi: { setTimeout: (callback, delay) => { assert.equal(delay, 1000); release = callback; } },
  });
  assert.equal(link.href, "blob:demod-export");
  assert.equal(link.download, "capture-mfsk.json");
  assert.equal(clicked, true);
  assert.equal(removed, true);
  assert.equal(revoked, null);
  release();
  assert.equal(revoked, "blob:demod-export");
});
