/** Build a reproducible export for protocol-independent symbol analysis. */
export function createDemodExport({ capture, mode, result, settings, createdAt = new Date().toISOString() }) {
  if (!capture || typeof capture.id !== "string" || typeof capture.sha256 !== "string") {
    throw new TypeError("capture id and SHA-256 are required");
  }
  if (typeof mode !== "string" || !mode || !result || typeof result.demodulator !== "string") {
    throw new TypeError("mode and demodulator result are required");
  }
  if (!settings || typeof settings !== "object") throw new TypeError("demodulator settings are required");
  return {
    schema_version: 1,
    artifact_type: "protocol-independent-demodulation",
    created_at: createdAt,
    modulation_hypothesis: mode,
    protocol: null,
    source_capture: {
      capture_id: capture.id,
      sha256: capture.sha256,
      sample_rate: capture.sample_rate,
      sample_count: capture.sample_count,
    },
    settings: { ...settings },
    result,
  };
}

export function downloadDemodExport(artifact, filename, { documentRef = document, urlApi = URL, timerApi = globalThis } = {}) {
  const blob = new Blob([`${JSON.stringify(artifact, null, 2)}\n`], { type: "application/json" });
  const url = urlApi.createObjectURL(blob);
  const link = documentRef.createElement("a");
  link.href = url;
  link.download = filename;
  documentRef.body.append(link);
  link.click();
  link.remove();
  timerApi.setTimeout(() => urlApi.revokeObjectURL(url), 1000);
}
