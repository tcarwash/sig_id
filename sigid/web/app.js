const el = (id) => document.getElementById(id);
const state = { capture: null, waterfall: null, drag: null, selection: null, regions: [] };
const baseCanvas = el("waterfall");
const overlayCanvas = el("overlay");
let livePoll = null;

async function refreshAudioDevices() {
  try {
    const devices = await request("/api/audio/devices");
    const select = el("audio-device");
    select.replaceChildren(new Option("Default input", ""));
    for (const device of devices) select.add(new Option(`${device.name} · ${device.max_input_channels} ch`, String(device.id)));
    el("live-status").textContent = devices.length ? `${devices.length} input device${devices.length === 1 ? "" : "s"} available` : "No audio inputs found";
  } catch (error) {
    el("live-status").textContent = error.message;
  }
}

el("live-start").addEventListener("click", async () => {
  clearError();
  try {
    const device = el("audio-device").value;
    const started = await request("/api/live/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: device === "" ? null : Number(device), sample_rate: Number(el("audio-rate").value), channels: 1, max_duration_seconds: 600 }),
    });
    el("live-start").disabled = true;
    el("live-stop").disabled = false;
    el("live-status").textContent = `Recording · ${started.device_name} · ${started.sample_rate.toLocaleString()} Hz`;
    livePoll = setInterval(async () => {
      try {
        const status = await request("/api/live");
        if (status.active) el("live-status").textContent = `${status.state} · ${status.frames_written.toLocaleString()} samples`;
      } catch (error) { showError(error.message); }
    }, 1000);
  } catch (error) { showError(error.message); }
});

el("live-stop").addEventListener("click", async () => {
  clearError();
  el("live-stop").disabled = true;
  try {
    const result = await request("/api/live/stop", { method: "POST" });
    state.capture = result.capture;
    await loadCapture();
    el("live-status").textContent = `Saved live capture · ${result.frames_written.toLocaleString()} samples`;
  } catch (error) { showError(error.message); }
  finally {
    if (livePoll) clearInterval(livePoll);
    livePoll = null;
    el("live-start").disabled = false;
    el("live-stop").disabled = true;
  }
});

refreshAudioDevices();

function showError(message) {
  const target = el("error");
  target.textContent = message;
  target.classList.remove("hidden");
}

function clearError() {
  el("error").classList.add("hidden");
}

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

el("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const file = el("wav-file").files[0];
  if (!file) return;
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "Loading…";
  try {
    const form = new FormData();
    form.append("file", file);
    state.capture = await request("/api/captures", { method: "POST", body: form });
    await loadCapture();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Load recording";
  }
});

async function loadCapture() {
  el("empty-state").classList.add("hidden");
  el("workspace").classList.remove("hidden");
  const c = state.capture;
  el("capture-title").textContent = c.source_metadata.uploaded_filename || c.source_metadata.device_name || "Audio recording";
  const duration = c.sample_count / c.sample_rate;
  el("capture-meta").textContent = `${c.sample_rate.toLocaleString()} samples/s · ${c.channels} channel${c.channels === 1 ? "" : "s"} · ${duration.toFixed(2)} seconds · ${c.source_kind}`;
  el("duration").textContent = `${duration.toFixed(2)} s`;
  el("audio-player").src = `/api/captures/${c.id}/audio`;
  el("waterfall-state").textContent = "Calculating spectrum…";
  state.waterfall = await request(`/api/captures/${c.id}/waterfall?fft_size=1024&hop_size=256&max_rows=640`);
  drawWaterfall();
  state.regions = await request(`/api/captures/${c.id}/regions`);
  renderRegions();
  el("waterfall-state").textContent = `${state.waterfall.db.length} spectrum rows · ${state.waterfall.fft_size}-point FFT`;
}

function colorFor(db, min, max) {
  const t = Math.max(0, Math.min(1, (db - min) / (max - min)));
  const stops = [[8, 17, 35], [19, 77, 135], [20, 166, 158], [210, 218, 121], [255, 244, 208]];
  const p = t * (stops.length - 1);
  const a = Math.floor(p);
  const b = Math.min(stops.length - 1, a + 1);
  const f = p - a;
  return [0, 1, 2].map((i) => Math.round(stops[a][i] * (1 - f) + stops[b][i] * f));
}

function drawWaterfall() {
  const data = state.waterfall;
  if (!data || !data.db.length) return;
  const bins = data.frequency_hz.length;
  const rows = data.db.length;
  baseCanvas.width = bins;
  baseCanvas.height = rows;
  overlayCanvas.width = bins;
  overlayCanvas.height = rows;
  const context = baseCanvas.getContext("2d", { alpha: false });
  const image = context.createImageData(bins, rows);
  const min = -95;
  const max = -20;
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < bins; x += 1) {
      const [r, g, b] = colorFor(data.db[y][x], min, max);
      const p = (y * bins + x) * 4;
      image.data[p] = r;
      image.data[p + 1] = g;
      image.data[p + 2] = b;
      image.data[p + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const nyquist = data.sample_rate / 2;
  el("freq-high").textContent = `${Math.round(nyquist).toLocaleString()} Hz`;
  el("freq-low").textContent = "0 Hz";
  drawOverlays();
}

function sampleToRow(sample) {
  const starts = state.waterfall.sample_starts;
  let lo = 0;
  let hi = starts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] < sample) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, Math.min(starts.length - 1, lo));
}

function drawOverlays() {
  const data = state.waterfall;
  if (!data) return;
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const nyquist = data.sample_rate / 2;
  const bins = data.frequency_hz.length;
  for (const region of state.regions) {
    const rowA = sampleToRow(region.sample_start);
    const rowB = Math.max(rowA + 1, sampleToRow(region.sample_start + region.sample_count));
    const low = region.frequency_low_hz ?? 0;
    const high = region.frequency_high_hz ?? nyquist;
    const x = Math.max(0, Math.floor((low / nyquist) * (bins - 1)));
    const width = Math.max(2, Math.ceil(((high - low) / nyquist) * (bins - 1)));
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, rowA, Math.min(width, bins - x), Math.max(2, rowB - rowA));
  }
  if (state.selection) {
    const { x1, y1, x2, y2 } = state.selection;
    ctx.strokeStyle = "#72fff0";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.max(2, Math.abs(x2 - x1)), Math.max(2, Math.abs(y2 - y1)));
    ctx.setLineDash([]);
  }
}

function canvasPoint(event) {
  const rect = overlayCanvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(overlayCanvas.width - 1, Math.round((event.clientX - rect.left) * overlayCanvas.width / rect.width))),
    y: Math.max(0, Math.min(overlayCanvas.height - 1, Math.round((event.clientY - rect.top) * overlayCanvas.height / rect.height))),
  };
}

overlayCanvas.addEventListener("pointerdown", (event) => {
  if (!state.waterfall) return;
  overlayCanvas.setPointerCapture(event.pointerId);
  const point = canvasPoint(event);
  state.drag = { ...point, x2: point.x, y2: point.y };
  state.selection = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
  el("hint").classList.add("hidden");
  drawOverlays();
});

overlayCanvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const point = canvasPoint(event);
  state.drag.x2 = point.x;
  state.drag.y2 = point.y;
  state.selection = { x1: state.drag.x, y1: state.drag.y, x2: point.x, y2: point.y };
  drawOverlays();
});

overlayCanvas.addEventListener("pointerup", (event) => {
  if (!state.drag) return;
  const point = canvasPoint(event);
  const data = state.waterfall;
  const rowStart = Math.min(state.drag.y, point.y);
  const rowEnd = Math.max(state.drag.y, point.y);
  const rowSamples = data.sample_starts;
  const sampleStart = rowSamples[rowStart];
  const sampleEnd = Math.min(data.sample_count, rowSamples[rowEnd] + data.fft_size);
  const nyquist = data.sample_rate / 2;
  const binStep = nyquist / (data.frequency_hz.length - 1);
  const xLow = Math.min(state.drag.x, point.x);
  const xHigh = Math.max(state.drag.x, point.x);
  state.selection = {
    sample_start: sampleStart,
    sample_count: Math.max(1, sampleEnd - sampleStart),
    frequency_low_hz: Math.max(0, xLow * binStep),
    frequency_high_hz: Math.min(nyquist, Math.max(xLow + 1, xHigh) * binStep),
    x1: state.drag.x, y1: state.drag.y, x2: point.x, y2: point.y,
  };
  state.drag = null;
  el("selection-summary").textContent = `${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz`;
  const t0 = sampleStart / data.sample_rate;
  const t1 = sampleEnd / data.sample_rate;
  el("selection-details").textContent = `Samples ${sampleStart.toLocaleString()}–${sampleEnd.toLocaleString()} · ${t0.toFixed(3)}–${t1.toFixed(3)} s`;
  el("save-region").disabled = false;
  drawOverlays();
});

el("save-region").addEventListener("click", async () => {
  if (!state.selection || !state.capture) return;
  clearError();
  const button = el("save-region");
  button.disabled = true;
  try {
    const selection = state.selection;
    await request(`/api/captures/${state.capture.id}/regions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection.sample_start,
        sample_count: selection.sample_count,
        frequency_low_hz: selection.frequency_low_hz,
        frequency_high_hz: selection.frequency_high_hz,
        family: el("family").value.trim() || null,
        method: el("method").value.trim() || null,
        protocol: el("protocol").value.trim() || null,
        variant: el("variant").value.trim() || null,
        notes: el("notes").value.trim(),
      }),
    });
    state.regions = await request(`/api/captures/${state.capture.id}/regions`);
    renderRegions();
    state.selection = null;
    el("selection-summary").textContent = "No region selected";
    el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
    for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = "";
    drawOverlays();
  } catch (error) {
    showError(error.message);
    button.disabled = false;
  }
});

function renderRegions() {
  const list = el("region-list");
  el("region-count").textContent = String(state.regions.length);
  list.replaceChildren();
  if (!state.regions.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = "No labeled regions in this recording yet.";
    list.append(empty);
  } else {
    for (const region of state.regions) {
      const row = document.createElement("div");
      row.className = "region-row";
      const main = document.createElement("div");
      main.className = "region-main";
      main.textContent = [region.family, region.method].filter(Boolean).join(" · ") || "Unknown signal";
      const detail = document.createElement("div");
      detail.className = "region-detail";
      detail.textContent = `${(region.sample_start / state.capture.sample_rate).toFixed(2)} s · ${(region.sample_count / state.capture.sample_rate).toFixed(2)} s duration`;
      const left = document.createElement("div");
      left.append(main, detail);
      const protocol = document.createElement("div");
      protocol.className = "region-protocol";
      protocol.textContent = region.protocol ? `${region.protocol}${region.variant ? ` / ${region.variant}` : ""}` : "Protocol unknown";
      const badge = document.createElement("span");
      badge.className = "region-badge";
      badge.textContent = region.review_state === "accepted" ? "HUMAN LABEL" : "PENDING";
      row.append(left, protocol, badge);
      list.append(row);
    }
  }
  drawOverlays();
}

el("wav-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) document.querySelector(".file-button").textContent = file.name;
});
