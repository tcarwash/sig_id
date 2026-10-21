/** Binary spectrum wire format: uint32 header length, padded JSON, row-major float32. */
export function decodeSpectrumPacket(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 4)
    throw new Error("Invalid spectrum packet");
  const size = new DataView(buffer).getUint32(0, true);
  if (size % 4 || size > 1_000_000 || size + 4 > buffer.byteLength)
    throw new Error("Invalid spectrum header");
  const metadata = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 4, size)),
  );
  const [rows, bins] = metadata.shape || [];
  if (
    metadata.schema_version !== 1 ||
    !Number.isSafeInteger(rows) ||
    !Number.isSafeInteger(bins) ||
    rows < 0 ||
    bins < 1 ||
    rows > 1024 ||
    bins > 4097 ||
    buffer.byteLength !== size + 4 + rows * bins * 4 ||
    metadata.sample_starts?.length !== rows ||
    metadata.frequency_hz?.length !== bins
  )
    throw new Error("Invalid spectrum dimensions");
  if (
    typeof metadata.session_id !== "string" ||
    !Number.isSafeInteger(metadata.sample_rate) ||
    metadata.sample_rate <= 0 ||
    !Number.isSafeInteger(metadata.sample_count) ||
    metadata.sample_count < 0 ||
    !Number.isSafeInteger(metadata.fft_size) ||
    metadata.fft_size < 32 ||
    !Number.isSafeInteger(metadata.hop_size) ||
    metadata.hop_size < 1 ||
    metadata.sample_starts.some(
      (sample, index) =>
        !Number.isSafeInteger(sample) ||
        sample < 0 ||
        sample + metadata.fft_size > metadata.sample_count ||
        (index > 0 && sample <= metadata.sample_starts[index - 1]),
    ) ||
    metadata.frequency_hz.some(
      (frequency, index) =>
        !Number.isFinite(frequency) ||
        frequency < 0 ||
        frequency > 3000 ||
        (index > 0 && frequency <= metadata.frequency_hz[index - 1]),
    )
  )
    throw new Error("Invalid spectrum coordinates");
  const values = new Float32Array(buffer, 4 + size);
  return {
    ...metadata,
    db: Array.from({ length: rows }, (_, i) =>
      values.subarray(i * bins, (i + 1) * bins),
    ),
  };
}

/** Immutable snapshots let a frozen viewport retain its data as the cache advances. */
export class LiveWaterfall {
  constructor(maxSeconds = 60) {
    if (!Number.isFinite(maxSeconds) || maxSeconds <= 0)
      throw new RangeError("Invalid history bound");
    this.maxSeconds = maxSeconds;
    this.data = null;
  }
  get lastSample() {
    return this.data?.sample_starts.at(-1) ?? -1;
  }
  append(packet) {
    const old = this.data;
    const same =
      old &&
      old.session_id === packet.session_id &&
      old.sample_rate === packet.sample_rate &&
      old.fft_size === packet.fft_size &&
      old.hop_size === packet.hop_size;
    if (same && packet.sample_count < old.sample_count) return false;
    const last = same ? this.lastSample : -1;
    const indices = packet.sample_starts
      .map((start, i) => [start, i])
      .filter(([start]) => start > last);
    if (same && packet.sample_count === old.sample_count && !indices.length)
      return false;
    const starts = [
      ...(same ? old.sample_starts : []),
      ...indices.map(([start]) => start),
    ];
    const db = [
      ...(same ? old.db : []),
      ...indices.map(([, index]) => packet.db[index]),
    ];
    const cutoff = packet.sample_count - packet.sample_rate * this.maxSeconds;
    let first = 0;
    while (first < starts.length && starts[first] < cutoff) first++;
    this.data = {
      ...packet,
      sample_starts: starts.slice(first),
      db: db.slice(first),
      gap_count:
        (same ? old.gap_count : 0) + Number(Boolean(packet.gap_before)),
    };
    return true;
  }
}

export function liveDisplayBounds(sampleCount, sampleRate, seconds = 30) {
  return {
    startSample: sampleCount - seconds * sampleRate,
    endSample: sampleCount,
  };
}
