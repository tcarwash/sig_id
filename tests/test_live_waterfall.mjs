import assert from "node:assert/strict";
import test from "node:test";
import {
  LiveWaterfall,
  decodeSpectrumPacket,
  liveDisplayBounds,
} from "../sigid/web/live-waterfall.mjs";
function packet(starts, total, session = "a") {
  return {
    session_id: session,
    sample_count: total,
    sample_rate: 100,
    fft_size: 32,
    hop_size: 10,
    frequency_hz: [0, 50],
    sample_starts: starts,
    db: starts.map((s) => new Float32Array([s, -80])),
  };
}
test("cache deduplicates and bounds rows while frozen snapshots remain immutable", () => {
  const cache = new LiveWaterfall(2);
  cache.append(packet([0, 10, 20], 30));
  const frozen = cache.data;
  cache.append(packet([10, 20, 30, 40], 50));
  assert.deepEqual(cache.data.sample_starts, [0, 10, 20, 30, 40]);
  assert.deepEqual(frozen.sample_starts, [0, 10, 20]);
  assert.equal(cache.append(packet([0], 10)), false);
  cache.append(packet([290, 300], 310));
  assert.deepEqual(cache.data.sample_starts, [290, 300]);
  cache.append(packet([0, 10], 20, "new"));
  assert.equal(cache.data.session_id, "new");
  assert.deepEqual(cache.data.sample_starts, [0, 10]);
});
test("startup display has a fixed thirty second window including blank prehistory", () => {
  assert.deepEqual(liveDisplayBounds(100, 100), {
    startSample: -2900,
    endSample: 100,
  });
  assert.deepEqual(liveDisplayBounds(5000, 100), {
    startSample: 2000,
    endSample: 5000,
  });
});
test("binary packet validates length, decodes rows and rejects corrupt framing", () => {
  const meta = { ...packet([0, 10], 50), schema_version: 1, shape: [2, 2] };
  delete meta.db;
  const raw = new TextEncoder().encode(JSON.stringify(meta));
  const size = Math.ceil(raw.length / 4) * 4;
  const buffer = new ArrayBuffer(4 + size + 16);
  new DataView(buffer).setUint32(0, size, true);
  new Uint8Array(buffer, 4, size).fill(32);
  new Uint8Array(buffer, 4, raw.length).set(raw);
  new Float32Array(buffer, 4 + size).set([-40, -50, -60, -70]);
  const decoded = decodeSpectrumPacket(buffer);
  assert.deepEqual([...decoded.db[1]], [-60, -70]);
  assert.throws(() => decodeSpectrumPacket(buffer.slice(0, -1)), /dimensions/);
  assert.throws(() => decodeSpectrumPacket(new ArrayBuffer(2)), /packet/);
});
