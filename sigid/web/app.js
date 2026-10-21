const el = (id) => document.getElementById(id);
const state = { capture: null, waterfall: null, drag: null, selection: null, regions: [], tracks: [], liveView: false, modelAvailable: false };
const baseCanvas = el("waterfall");
const overlayCanvas = el("overlay");
let livePoll = null;
let fldigiPoll = null;
let fldigiOffset = null;
let trialPoll = null;
let watchedTrialTrack = null;

async function refreshFldigiStatus() {
  try {
    const status = await request("/api/fldigi/status");
    const label = el("fldigi-status");
    if (!status.connected) {
      label.textContent = `Fldigi disconnected · ${status.error || "start Fldigi with its XML-RPC server enabled"}`;
      el("connection").lastChild.textContent = " Decoder offline";
      el("fldigi-mode").disabled = true;
      el("fldigi-select").disabled = true;
      return false;
    }
    label.textContent = `${status.name} ${status.version} · ${status.current_mode} · ${status.mode_count} modes · audio route not verified`;
    el("connection").lastChild.textContent = ` ${status.name} receive connected`;
    if (fldigiOffset === null) fldigiOffset = status.receive_text_length;
    const modes = await request("/api/fldigi/modes");
    const select = el("fldigi-mode");
    const previous = select.value;
    select.replaceChildren(...modes.map((name) => new Option(name, name)));
    if (modes.includes(previous)) select.value = previous;
    else select.value = status.current_mode;
    select.disabled = false;
    el("fldigi-select").disabled = false;
    return true;
  } catch (error) {
    el("fldigi-status").textContent = `Fldigi unavailable · ${error.message}`;
    return false;
  }
}

async function pollFldigiOutput() {
  if (trialPoll) return;
  try {
    const result = await request(`/api/fldigi/output?start=${fldigiOffset}&limit=8192`);
    if (result.end < fldigiOffset) fldigiOffset = 0;
    fldigiOffset = result.end;
    if (result.text) {
      const target = el("decoded-text");
      const old = target.textContent === "Decoded text will appear here." ? "" : target.textContent;
      target.textContent = (old + result.text).slice(-20000);
      target.scrollTop = target.scrollHeight;
    }
  } catch (error) {
    el("fldigi-status").textContent = `Fldigi output unavailable · ${error.message}`;
    if (fldigiPoll) clearInterval(fldigiPoll);
    fldigiPoll = setTimeout(watchFldigi, 5000);
  }
}

function watchDecoderTrial(result) {
  if (trialPoll || watchedTrialTrack === result.track_id) return;
  watchedTrialTrack = result.track_id;
  fldigiOffset = result.output_offset;
  el("fldigi-status").textContent = `Trying ${result.current_mode} for ${result.track_id} · receive only · audio routing is external`;
  el("decoded-text").textContent = `Automatic receive trial: ${result.current_mode}\nTentative only · waiting ${result.note}`;
  trialPoll = setInterval(async () => {
    try {
      const trial = await request("/api/fldigi/trials/current");
      el("fldigi-status").textContent = `${trial.status} · ${trial.attempted_modes.join(" → ")} · ${trial.note}`;
      if (trial.status === "tentative_output") {
        el("decoded-text").textContent = `TENTATIVE · ${trial.current_mode}\n${trial.text}`;
        fldigiOffset = trial.output_end ?? fldigiOffset;
      } else if (trial.status !== "running") {
        el("decoded-text").textContent = `${trial.status.toUpperCase()} · ${trial.current_mode || "no mode"}\n${trial.note}`;
        if (trialPoll) clearInterval(trialPoll);
        trialPoll = null;
        pollFldigiOutput();
      }
    } catch (error) {
      el("fldigi-status").textContent = `Automatic trial stopped · ${error.message}`;
      if (trialPoll) clearInterval(trialPoll);
      trialPoll = null;
    }
  }, 1000);
}

async function startDecoderTrial(track) {
  if (!track.track_id || !state.liveView) return;
  try {
    const result = await request("/api/fldigi/trials/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track_id: track.track_id,
        modulation_hypothesis: track.model_proposal?.available && !track.model_proposal.abstained
          ? track.model_proposal.family
          : track.modulation_hypothesis,
      }),
    });
    watchDecoderTrial(result);
  } catch (error) { showError(error.message); }
}

el("fldigi-select").addEventListener("click", async () => {
  try {
    const result = await request("/api/fldigi/mode", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: el("fldigi-mode").value }),
    });
    el("fldigi-status").textContent = `Selected ${result.current_mode} · audio routing not verified`;
    fldigiOffset = result.output_offset;
    el("decoded-text").textContent = "Decoder mode changed. Waiting for receive output…";
  } catch (error) { showError(error.message); }
});

async function watchFldigi() {
  if (await refreshFldigiStatus()) {
    if (fldigiPoll) clearInterval(fldigiPoll);
    pollFldigiOutput();
    fldigiPoll = setInterval(pollFldigiOutput, 1000);
  } else {
    fldigiPoll = setTimeout(watchFldigi, 5000);
  }
}
watchFldigi();

async function refreshModelStatus() {
  try {
    const model = await request("/api/model/status");
    state.modelAvailable = model.available;
    el("auto-trial").disabled = !model.available;
    el("model-status").textContent = model.available
      ? `Method model ready · family threshold ${(100 * model.family_threshold).toFixed(0)}% · method ${(100 * model.threshold).toFixed(0)}% · synthetic only`
      : `No model loaded${model.error ? ` · ${model.error}` : " · configure SIG_ID_MODEL_PATH"}`;
    updateModelButton();
  } catch (error) {
    el("auto-trial").disabled = true;
    el("model-status").textContent = `Model status unavailable · ${error.message}`;
  }
}
refreshModelStatus();

function updateModelButton() {
  el("model-propose").disabled = !state.modelAvailable || !state.capture || !state.selection || state.liveView;
}

el("model-propose").addEventListener("click", async () => {
  if (!state.capture || !state.selection || state.liveView || !state.modelAvailable) return;
  const selection = state.selection;
  const button = el("model-propose");
  button.disabled = true;
  button.textContent = "Analyzing…";
  el("label-state").textContent = "Calculating a provisional model proposal…";
  try {
    const result = await request(`/api/captures/${state.capture.id}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample_start: selection.sample_start, sample_count: Math.min(1_000_000, selection.sample_count) }),
    });
    if (!result.family && !result.method) {
      el("label-state").textContent = `Model abstained · top guess ${result.top_candidate} at ${(100 * result.confidence).toFixed(1)}%`;
    } else {
      el("family").value = result.family || "";
      el("method").value = result.method || "";
      el("protocol").value = "";
      el("variant").value = "";
      const evidence = `Model proposal ${result.method || result.family} (${(100 * (result.method ? result.confidence : result.family_confidence)).toFixed(1)}%; calibrated threshold). Review before saving; protocol remains unknown.`;
      el("notes").value = [el("notes").value.trim(), evidence].filter(Boolean).join(" ");
      el("label-state").textContent = `Proposed ${result.method || result.family} · review then save to accept`;
    }
  } catch (error) {
    el("label-state").textContent = `Model proposal failed · ${error.message}`;
  } finally {
    button.textContent = "Suggest method";
    updateModelButton();
  }
});

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
    state.capture = null;
    state.liveView = true;
    state.waterfall = null;
    state.selection = null;
    state.regions = [];
    state.tracks = [];
    baseCanvas.getContext("2d").clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    overlayCanvas.getContext("2d").clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    el("demod-run").disabled = true;
    el("mfsk-run").disabled = true;
    el("bpsk-run").disabled = true;
    el("qpsk-run").disabled = true;
    el("demod-output").textContent = "Stop and save the capture before running protocol-independent demodulation.";
    el("mfsk-output").textContent = el("demod-output").textContent;
    el("bpsk-output").textContent = el("demod-output").textContent;
    el("qpsk-output").textContent = el("demod-output").textContent;
    el("empty-state").classList.add("hidden");
    el("workspace").classList.remove("hidden");
    el("capture-title").textContent = "Live passband";
    el("capture-meta").textContent = `${started.sample_rate.toLocaleString()} samples/s · ${started.device_name} · rolling analysis`;
    el("time-start").textContent = "−10 s";
    el("time-end").textContent = "Now";
    el("audio-player").removeAttribute("src");
    el("waterfall-state").textContent = "Waiting for audio frames…";
    el("track-status").textContent = "Waiting for enough samples to analyze…";
    el("save-region").disabled = true;
    updateModelButton();
    renderTracks();
    el("live-status").textContent = `Recording · ${started.device_name} · ${started.sample_rate.toLocaleString()} Hz`;
    renderRegions();
    livePoll = setInterval(async () => {
      try {
        const status = await request("/api/live");
        if (status.active) el("live-status").textContent = `${status.state} · ${status.frames_written.toLocaleString()} samples`;
        const autoTrial = el("auto-trial").checked;
        const analysis = await request(`/api/live/analysis?fft_size=256&hop_size=64&max_rows=640&auto_trial=${autoTrial}`);
        if (analysis.warming_up) {
          el("waterfall-state").textContent = `Collecting analysis window · ${analysis.window_sample_count.toLocaleString()} samples`;
          return;
        }
        state.waterfall = analysis.spectrum;
        state.tracks = analysis.tracks;
        if (analysis.automatic_trial?.status === "running") watchDecoderTrial(analysis.automatic_trial);
        el("waterfall-state").textContent = `${analysis.spectrum.db.length} rolling spectrum rows · latest 10 seconds`;
        el("track-status").textContent = state.modelAvailable
          ? "Model-ranked live candidates · proposals remain provisional"
          : "Live spectral heuristic · candidates are provisional";
        drawWaterfall();
        renderTracks();
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

el("synthetic-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  const wav = el("synthetic-wav").files[0];
  const manifest = el("synthetic-manifest").files[0];
  if (!wav || !manifest) return;
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "Validating…";
  try {
    const form = new FormData();
    form.append("wav_file", wav);
    form.append("manifest_file", manifest);
    const imported = await request("/api/synthetic/import", { method: "POST", body: form });
    state.capture = imported.capture;
    await loadCapture();
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Import labeled fixture";
  }
});

async function loadCapture() {
  state.liveView = false;
  state.selection = null;
  updateModelButton();
  el("demod-run").disabled = false;
  el("mfsk-run").disabled = false;
  el("bpsk-run").disabled = false;
  el("qpsk-run").disabled = false;
  el("empty-state").classList.add("hidden");
  el("workspace").classList.remove("hidden");
  state.tracks = [];
  renderTracks();
  const c = state.capture;
  el("capture-title").textContent = c.source_metadata.uploaded_filename || c.source_metadata.device_name || c.source_metadata.generator || "Audio recording";
  const manifestLink = el("manifest-download");
  if (c.source_kind === "synthetic" && c.source_metadata.manifest_file) {
    manifestLink.href = `/api/captures/${c.id}/manifest`;
    manifestLink.classList.remove("hidden");
  } else {
    manifestLink.classList.add("hidden");
  }
  const duration = c.sample_count / c.sample_rate;
  el("capture-meta").textContent = `${c.sample_rate.toLocaleString()} samples/s · ${c.channels} channel${c.channels === 1 ? "" : "s"} · ${duration.toFixed(2)} seconds · ${c.source_kind}`;
  el("time-start").textContent = "0 s";
  el("time-end").textContent = `${duration.toFixed(2)} s`;
  el("audio-player").src = `/api/captures/${c.id}/audio`;
  el("waterfall-state").textContent = "Calculating spectrum…";
  state.waterfall = await request(`/api/captures/${c.id}/waterfall?fft_size=1024&hop_size=256&max_rows=640`);
  drawWaterfall();
  state.regions = await request(`/api/captures/${c.id}/regions`);
  renderRegions();
  try {
    const result = await request(`/api/captures/${c.id}/tracks`);
    state.tracks = result.tracks;
    el("track-status").textContent = `Spectral heuristic · ${result.analysis.fft_size}-point FFT · candidates are provisional`;
  } catch (error) {
    state.tracks = [];
    el("track-status").textContent = `Candidate analysis unavailable · ${error.message}`;
  }
  renderTracks();
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
  const firstFrequency = data.frequency_hz[0];
  const lastFrequency = data.frequency_hz[bins - 1];
  el("freq-low").textContent = `${Math.round(firstFrequency).toLocaleString()} Hz`;
  el("freq-high").textContent = `${Math.round(lastFrequency).toLocaleString()} Hz`;
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
  for (const track of state.tracks) {
    const rowA = sampleToRow(track.sample_start);
    const rowB = Math.max(rowA + 1, sampleToRow(track.sample_start + track.sample_count));
    const x = Math.max(0, Math.floor((track.frequency_low_hz / nyquist) * (bins - 1)));
    const right = Math.min(bins - 1, Math.ceil((track.frequency_high_hz / nyquist) * (bins - 1)));
    ctx.strokeStyle = track.track_state === "persistent" ? "#52d6d0" : "#f1be69";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(x, rowA, Math.max(2, right - x), Math.max(2, rowB - rowA));
  }
  ctx.setLineDash([]);
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

function selectTrack(track) {
  const data = state.waterfall;
  if (!data) return;
  const nyquist = data.sample_rate / 2;
  const bins = data.frequency_hz.length;
  const x1 = Math.max(0, Math.min(bins - 1, Math.round(track.frequency_low_hz / nyquist * (bins - 1))));
  const x2 = Math.max(x1 + 1, Math.min(bins - 1, Math.round(track.frequency_high_hz / nyquist * (bins - 1))));
  const y1 = sampleToRow(track.sample_start);
  const y2 = Math.max(y1 + 1, sampleToRow(track.sample_start + track.sample_count));
  const binStep = nyquist / (bins - 1);
  const sampleEnd = Math.min(data.sample_count, track.sample_start + track.sample_count);
  state.selection = { sample_start: track.sample_start, sample_count: Math.max(1, sampleEnd - track.sample_start), frequency_low_hz: x1 * binStep, frequency_high_hz: x2 * binStep, x1, y1, x2, y2 };
  el("selection-summary").textContent = `${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz · ${track.modulation_hypothesis}`;
  el("selection-details").textContent = `Provisional candidate · ${(track.time_start_seconds).toFixed(2)}–${track.time_end_seconds.toFixed(2)} s · ${track.tone_count_candidate} tone centers · protocol unknown`;
  el("family").value = "";
  el("method").value = "";
  el("protocol").value = "";
  el("variant").value = "";
  const trackEvidence = track.track_id ? ` Track ${track.track_id} observed ${track.track_observations} times (${track.track_state}).` : "";
  el("notes").value = `Auto-detected candidate (${track.modulation_hypothesis}); review before labeling.${trackEvidence} SNR ${track.peak_snr_db.toFixed(1)} dB.`;
  el("save-region").disabled = state.liveView || !state.capture;
  updateModelButton();
  el("hint").classList.add("hidden");
  drawOverlays();
  el("selection-summary").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderTracks() {
  const list = el("track-list");
  el("track-count").textContent = String(state.tracks.length);
  list.replaceChildren();
  if (!state.tracks.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = "No persistent narrow-tone candidates detected. This heuristic can miss signals; manual selection is always available.";
    list.append(empty);
  } else {
    for (const track of state.tracks) {
      const row = document.createElement("div");
      row.className = "track-row";
      const summary = document.createElement("div");
      summary.className = "track-summary";
      const identity = track.track_id ? `${track.track_id.split(":").at(-1)} · ${track.track_state}` : "single analysis";
      const proposal = track.model_proposal;
      const modelLabel = proposal?.available
        ? proposal.abstained
          ? ` · model abstained (${proposal.top_candidate}, ${(100 * proposal.confidence).toFixed(0)}%)`
          : ` · model ${proposal.method_id} ${(100 * proposal.confidence).toFixed(0)}%`
        : "";
      summary.textContent = `${identity} · ${track.modulation_hypothesis} · ${track.tone_count_candidate} tone centers${modelLabel}`;
      const detail = document.createElement("div");
      detail.className = "region-detail";
      detail.textContent = `${track.frequency_low_hz.toFixed(0)}–${track.frequency_high_hz.toFixed(0)} Hz · ${track.time_start_seconds.toFixed(2)}–${track.time_end_seconds.toFixed(2)} s · ${track.peak_snr_db.toFixed(1)} dB peak SNR`;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Review region";
      button.addEventListener("click", () => selectTrack(track));
      row.append(summary, detail, button);
      if (state.liveView && track.track_id) {
        const tryButton = document.createElement("button");
        tryButton.className = "track-auto-button";
        tryButton.type = "button";
        tryButton.textContent = "Try tested modes";
        tryButton.addEventListener("click", () => startDecoderTrial(track));
        row.append(tryButton);
      }
      list.append(row);
    }
  }
  drawOverlays();
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
  el("save-region").disabled = state.liveView || !state.capture;
  updateModelButton();
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
    updateModelButton();
    el("selection-summary").textContent = "No region selected";
    el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
    for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = "";
    drawOverlays();
  } catch (error) {
    showError(error.message);
    button.disabled = false;
  }
});

el("demod-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  clearError();
  const button = el("demod-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("demod-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const result = await request(`/api/captures/${state.capture.id}/demod/2fsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection?.sample_start ?? 0,
        sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
        symbol_rate: Number(el("demod-rate").value),
        tone_0_hz: Number(el("demod-tone0").value),
        tone_1_hz: Number(el("demod-tone1").value),
        timing_offset_samples: Number(el("demod-offset").value),
      }),
    });
    const confidence = result.soft_metrics.length
      ? result.soft_metrics.reduce((sum, value) => sum + Math.abs(value), 0) / result.soft_metrics.length
      : 0;
    const groupedBits = result.bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    el("demod-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols · mean |soft metric| ${confidence.toFixed(3)}\nProtocol: unknown\n\n${groupedBits}`;
  } catch (error) {
    el("demod-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover raw bits";
  }
});

el("mfsk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  clearError();
  const button = el("mfsk-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("mfsk-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const result = await request(`/api/captures/${state.capture.id}/demod/mfsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection?.sample_start ?? 0,
        sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
        symbol_rate: Number(el("mfsk-rate").value),
        tone_count: Number(el("mfsk-count").value),
        base_tone_hz: Number(el("mfsk-base").value),
        tone_spacing_hz: Number(el("mfsk-spacing").value),
        timing_offset_samples: Number(el("mfsk-offset").value),
      }),
    });
    const confidence = result.soft_confidence.length
      ? result.soft_confidence.reduce((sum, value) => sum + value, 0) / result.soft_confidence.length
      : 0;
    const groupedSymbols = result.symbols.join(" ");
    const bits = result.bits === null ? "Bits: unavailable for non-power-of-two tone count" : `Bits:\n${result.bits.match(/.{1,64}/g)?.join("\n") || ""}`;
    el("mfsk-output").textContent = `${result.demodulator} · ${result.tone_count} tones · ${result.symbol_count} symbols · mean soft confidence ${confidence.toFixed(3)}\nProtocol: unknown\n\nTone indices:\n${groupedSymbols}\n\n${bits}`;
  } catch (error) {
    el("mfsk-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover tone symbols";
  }
});

el("bpsk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  clearError();
  const button = el("bpsk-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("bpsk-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const result = await request(`/api/captures/${state.capture.id}/demod/bpsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection?.sample_start ?? 0,
        sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
        symbol_rate: Number(el("bpsk-rate").value),
        carrier_hz: Number(el("bpsk-carrier").value),
        timing_offset_samples: Number(el("bpsk-offset").value),
      }),
    });
    const grouped = (bits) => bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    const confidence = result.soft_metrics.length
      ? result.soft_metrics.reduce((sum, value) => sum + Math.abs(value), 0) / result.soft_metrics.length
      : 0;
    el("bpsk-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols · mean |soft metric| ${confidence.toFixed(3)}\nPolarity ambiguous: ${result.polarity_ambiguous}\nProtocol: unknown\n\nBit hypothesis A:\n${grouped(result.bits)}\n\nBit hypothesis B (inverted):\n${grouped(result.inverted_bits)}`;
  } catch (error) {
    el("bpsk-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover BPSK bits";
  }
});

el("qpsk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  clearError();
  const button = el("qpsk-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("qpsk-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const result = await request(`/api/captures/${state.capture.id}/demod/qpsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection?.sample_start ?? 0,
        sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
        symbol_rate: Number(el("qpsk-rate").value),
        carrier_hz: Number(el("qpsk-carrier").value),
        timing_offset_samples: Number(el("qpsk-offset").value),
      }),
    });
    const grouped = (bits) => bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    const hypotheses = result.symbol_hypotheses.map((item, index) =>
      `Hypothesis ${index + 1} · rotation ${(item.rotation_radians * 180 / Math.PI).toFixed(0)}°\nSymbols: ${item.symbols.join(" ")}\nBits:\n${grouped(item.bits)}`
    ).join("\n\n");
    el("qpsk-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols · absolute rotation unknown\nProtocol: unknown\n\n${hypotheses}`;
  } catch (error) {
    el("qpsk-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover QPSK symbols";
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
      const badgeLabels = { synthetic: "SYNTHETIC", human: "HUMAN LABEL", decoder: "DECODER", external: "EXTERNAL", rsid: "RSID", prediction: "PREDICTION", unknown: "UNKNOWN" };
      badge.textContent = region.review_state === "accepted" ? (badgeLabels[region.source] || "ACCEPTED") : "PENDING REVIEW";
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

for (const [inputId, labelText] of [["synthetic-wav", "Choose fixture WAV"], ["synthetic-manifest", "Choose manifest"]]) {
  el(inputId).addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) document.querySelector(`label[for="${inputId}"]`).textContent = file.name;
    else document.querySelector(`label[for="${inputId}"]`).textContent = labelText;
  });
}
