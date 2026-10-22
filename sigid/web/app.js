import { liveTimeLabels } from "/waterfall-window.mjs";
import { decodeSpectrumPacket, LiveWaterfall, liveDisplayBounds } from "/live-waterfall.mjs";
import { createDemodExport, downloadDemodExport } from "/demod-export.mjs";
import { layoutSignalLabels, signalCaption, hitSignal } from "/signal-overlays.mjs";

const el = (id) => document.getElementById(id);
const state = { capture: null, waterfall: null, drag: null, selection: null, regions: [], tracks: [], liveView: false, modelAvailable: false, view: null, demodExports: {}, selectedTrack: null, frozen: false };
const baseCanvas = el("waterfall");
const overlayCanvas = el("overlay");
let livePoll = null;
let spectrumPoll = null;
let liveDisplay = null;
let fldigiPoll = null;
let fldigiOffset = null;
let trialPoll = null;
let watchedTrialTrack = null;

const DEMOD_EXPORT_BUTTONS = {
  "2fsk": "demod-export-2fsk",
  ook: "demod-export-ook",
  mfsk: "demod-export-mfsk",
  bpsk: "demod-export-bpsk",
  qpsk: "demod-export-qpsk",
};

function clearDemodExports() {
  state.demodExports = {};
  for (const id of Object.values(DEMOD_EXPORT_BUTTONS)) el(id).disabled = true;
}

function storeDemodExport(mode, result, settings) {
  state.demodExports[mode] = createDemodExport({ capture: state.capture, mode, result, settings });
  el(DEMOD_EXPORT_BUTTONS[mode]).disabled = false;
  showRawTool(mode);
  showOutput("raw");
}

for (const [mode, buttonId] of Object.entries(DEMOD_EXPORT_BUTTONS)) {
  el(buttonId).addEventListener("click", () => {
    const artifact = state.demodExports[mode];
    if (!artifact) return;
    downloadDemodExport(artifact, `${artifact.source_capture.capture_id}-${mode}-demod.json`);
  });
}

async function refreshFldigiStatus() {
  try {
    const status = await request("/api/fldigi/status");
    const label = el("fldigi-status");
    if (!status.connected) {
      label.textContent = `Fldigi disconnected · ${status.error || "start Fldigi with its XML-RPC server enabled"}`;
      el("connection").lastChild.textContent = " Decoder offline";
      el("fldigi-mode").disabled = true;
      el("fldigi-select").disabled = true;
      el("fldigi-check-wefax").disabled = true;
      updateDecoderSelectionControls();
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
    el("fldigi-check-wefax").disabled = !status.wefax_output_available;
    el("wefax-status").textContent = status.wefax_output_available
      ? "WEFAX image reception available · poll after image reception completes"
      : "This decoder backend does not expose WEFAX image output";
    await refreshReceivedImages();
    updateDecoderSelectionControls();
    return true;
  } catch (error) {
    el("fldigi-status").textContent = `Fldigi unavailable · ${error.message}`;
    updateDecoderSelectionControls();
    return false;
  }
}

async function refreshReceivedImages(products = null) {
  const target = el("received-images");
  try {
    const items = products || await request("/api/fldigi/products");
    target.replaceChildren();
    if (!items.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No received image products yet.";
      target.append(empty);
      return;
    }
    for (const product of items) {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = product.image_url;
      image.alt = `Received WEFAX image ${product.id}`;
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      caption.textContent = `${product.source_filename} · ${(product.size_bytes / 1024).toFixed(1)} KiB · SHA-256 ${product.sha256.slice(0, 12)}…`;
      figure.append(image, caption);
      target.append(figure);
    }
  } catch (error) {
    el("wefax-status").textContent = `Could not load received images · ${error.message}`;
  }
}

el("fldigi-check-wefax").addEventListener("click", async () => {
  el("fldigi-check-wefax").disabled = true;
  try {
    const result = await request("/api/fldigi/products/wefax/poll?timeout_seconds=0", { method: "POST" });
    el("wefax-status").textContent = result.received
      ? `Imported received image · SHA-256 ${result.product.sha256}`
      : "No completed WEFAX image is waiting.";
    await refreshReceivedImages(result.products);
  } catch (error) {
    el("wefax-status").textContent = `WEFAX product check failed · ${error.message}`;
  } finally {
    el("fldigi-check-wefax").disabled = false;
  }
});

async function pollFldigiOutput() {
  if (trialPoll) return;
  const generation = state.outputGeneration;
  try {
    if (fldigiOffset === null) {
      const status = await request("/api/fldigi/status");
      if (generation === state.outputGeneration && status.connected) fldigiOffset = status.receive_text_length;
      return;
    }
    const result = await request(`/api/fldigi/output?start=${fldigiOffset}&limit=8192`);
    if (generation !== state.outputGeneration) return;
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
  const generation = state.outputGeneration;
  state.decoderTrial = {track_id:result.track_id,status:result.status,mode:result.current_mode,run_id:result.run_id};
  state.textOutputContext = `Saved receive run ${result.run_id?.slice(0,8) || "pending"} · track ${result.track_id} · tentative · shared audio input`;
  updateOutputContext();
  fldigiOffset = result.output_offset;
  el("fldigi-status").textContent = `Trying ${result.current_mode} for ${result.track_id} · receive only · audio routing is external`;
  el("decoded-text").textContent = `Automatic receive trial: ${result.current_mode}\nTentative only · waiting ${result.note}`;
  trialPoll = setInterval(async () => {
    try {
      const trial = await request("/api/fldigi/trials/current");
      if (generation !== state.outputGeneration) return;
      state.decoderTrial = {track_id:trial.track_id,status:trial.status,mode:trial.current_mode,run_id:trial.run_id};
      if (!state.frozen) renderSignalLabels();
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
        sample_start: track.sample_start,
        sample_count: track.sample_count,
        modulation_hypothesis: track.model_proposal?.available && !track.model_proposal.abstained
          ? track.model_proposal.family
          : track.modulation_hypothesis,
      }),
    });
    watchDecoderTrial(result);
  } catch (error) { showError(error.message); }
}

el("decode-selection").addEventListener("click", async () => {
  const selection = state.selection;
  const captureId = state.capture?.id;
  const mode = el("fldigi-mode").value;
  if (!selection || !captureId || state.liveView || !mode) return;
  const button = el("decode-selection");
  button.disabled = true;
  el("replay-status").textContent = `Queuing ${mode} replay for ${(selection.sample_start / state.waterfall.sample_rate).toFixed(2)}–${((selection.sample_start + selection.sample_count) / state.waterfall.sample_rate).toFixed(2)} s…`;
  try {
    const result = await request(`/api/captures/${captureId}/decode`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        mode,
        sample_start: selection.sample_start,
        sample_count: selection.sample_count,
        track_id: state.selectedTrack?.track_id || null,
        carrier_hz: Number.isFinite(selection.frequency_low_hz) && Number.isFinite(selection.frequency_high_hz)
          ? (selection.frequency_low_hz + selection.frequency_high_hz) / 2 : null,
      }),
    });
    el("replay-status").textContent = `Replay ${result.status} · ${mode} · isolated region input`;
    pollSavedDecodeRun(captureId, result.run_id);
  } catch (error) {
    el("replay-status").textContent = `Replay could not start · ${error.message}`;
  } finally {
    updateDecoderSelectionControls();
  }
});

async function pollSavedDecodeRun(captureId, runId) {
  for (let attempt = 0; attempt < 240; attempt++) {
    if (state.capture?.id !== captureId) return;
    try {
      const runs = await request(`/api/captures/${captureId}/decode-runs`);
      if (state.capture?.id !== captureId) return;
      state.decodeRuns = runs;
      renderDecodeRuns();
      const run = runs.find((item) => item.id === runId);
      if (run && !["queued", "running"].includes(run.status)) {
        el("replay-status").textContent = `${run.status.toUpperCase()} · ${run.current_mode} · ${run.note}`;
        if (run.output_text) {
          showOutput("text");
          state.textOutputContext = `Isolated replay ${run.id.slice(0,8)} · exact selected region · tentative text`;
          el("output-context").textContent = state.textOutputContext;
          el("decoded-text").textContent = run.output_text;
        }
        updateDecoderSelectionControls();
        return;
      }
    } catch (error) {
      el("replay-status").textContent = `Replay status unavailable · ${error.message}`;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  el("replay-status").textContent = "Replay is still running. Reopen this capture later to review its saved result.";
}

el("tag-as-mode").addEventListener("click", () => {
  const mode = el("fldigi-mode").value;
  if (!state.selection || !state.capture || state.liveView || !mode) return;
  el("variant").value = mode;
  el("label-state").textContent = `Saving ${mode} as an accepted human-reviewed mode label.`;
  openInspector("label");
  el("save-region").click();
});

el("fldigi-select").addEventListener("click", async () => {
  try {
    const result = await request("/api/fldigi/mode", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: el("fldigi-mode").value }),
    });
    el("fldigi-status").textContent = `Selected ${result.current_mode} · audio routing not verified`;
    fldigiOffset = result.output_offset;
    state.textOutputContext = `Shared Fldigi output · ${result.current_mode} · external audio route`;
    updateOutputContext();
    el("decoded-text").textContent = "Decoder mode changed. Waiting for receive output…";
    updateDecoderSelectionControls();
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
    const unknownStatus = model.unknown_rejection_validated
      ? ` · held-out ${model.unknown_rejection_scope} rejection`
      : " · unknown rejection unvalidated";
    el("model-status").textContent = model.available
      ? `Method model ready · method ${(100 * model.threshold).toFixed(0)}% · family ${(100 * model.family_threshold).toFixed(0)}%${unknownStatus} · synthetic only`
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
  const sourceCapture = state.capture;
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
    if (state.capture !== sourceCapture || state.selection !== selection) return;
    if (!result.family && !result.method) {
      el("label-state").textContent = result.novelty_rejected
        ? `Model abstained · outside trained modulation range (distance ${result.novelty_distance.toFixed(2)} > ${result.novelty_threshold.toFixed(2)})`
        : `Model abstained · top guess ${result.top_candidate} at ${(100 * result.confidence).toFixed(1)}%`;
    } else {
      el("family").value = result.family || "";
      el("method").value = result.method || "";
      el("protocol").value = "";
      el("variant").value = "";
      const noveltyEvidence = Number.isFinite(result.novelty_distance)
        ? ` Nearest known-mode distance ${result.novelty_distance.toFixed(2)} (limit ${result.novelty_threshold.toFixed(2)}).`
        : "";
      const evidence = `Model proposal ${result.method || result.family} (${(100 * (result.method ? result.confidence : result.family_confidence)).toFixed(1)}%; calibrated threshold).${noveltyEvidence} Review before saving; protocol remains unknown.`;
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
    clearDemodExports();
    state.capture = null;
    state.liveView = true;
    resetOutputSession();
    state.waterfall = null;
    state.view = null;
    state.selection = null;
    state.regions = [];
    state.tracks = [];
    baseCanvas.getContext("2d").clearRect(0, 0, baseCanvas.width, baseCanvas.height);
    overlayCanvas.getContext("2d").clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    el("demod-run").disabled = true;
    el("ook-run").disabled = true;
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
    el("time-start").textContent = "Now";
    el("time-end").textContent = "−30 s";
    el("audio-player").removeAttribute("src");
    el("waterfall-state").textContent = "Waiting for audio frames…";
    el("track-status").textContent = "Waiting for enough samples to analyze…";
    el("save-region").disabled = true;
    updateModelButton();
    renderTracks();
    el("live-status").textContent = `Recording · ${started.device_name} · ${started.sample_rate.toLocaleString()} Hz`;
    renderRegions();
    const generation = state.liveGeneration = (state.liveGeneration || 0) + 1;
    liveDisplay = new LiveWaterfall(60);
    startSpectrumStream(generation, started.session_id);
    let pending = false;
    livePoll = setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const status = await request("/api/live");
        if (generation !== state.liveGeneration || !state.liveView) return;
        if (status.active) el("live-status").textContent = state.frozen ? "Recording · view frozen" : `${status.state} · ${status.frames_written.toLocaleString()} samples`;
        const autoTrial = el("auto-trial").checked;
        const analysis = await request(`/api/live/analysis?fft_size=4096&hop_size=512&max_rows=640&include_spectrum=false&auto_trial=${autoTrial}`);
        if (generation !== state.liveGeneration || !state.liveView) return;
        if (analysis.automatic_trial?.status === "running") watchDecoderTrial(analysis.automatic_trial);
        if (state.frozen) return;
        if (analysis.warming_up) {
          el("waterfall-state").textContent = `Collecting analysis window · ${analysis.window_sample_count.toLocaleString()} samples`;
          return;
        }
        state.tracks = analysis.tracks;
        el("track-status").textContent = state.modelAvailable
          ? "Model-ranked live candidates · proposals remain provisional"
          : "Live spectral heuristic · candidates are provisional";
        drawWaterfall();
        renderTracks();
      } catch (error) { if (generation === state.liveGeneration) showError(error.message); }
      finally { pending = false; }
    }, 1000);
  } catch (error) { showError(error.message); }
});

el("live-stop").addEventListener("click", async () => {
  clearError();
  el("live-stop").disabled = true;
  try {
    const result = await request("/api/live/stop", { method: "POST" });
    state.liveGeneration = (state.liveGeneration || 0) + 1;
    if (livePoll) clearInterval(livePoll);
    if (spectrumPoll) clearTimeout(spectrumPoll);
    state.capture = result.capture;
    await loadCapture();
    el("live-status").textContent = result.truncated_reason
      ? `Saved partial live capture · ${result.truncated_reason} · ${result.frames_written.toLocaleString()} samples`
      : `Saved live capture · ${result.frames_written.toLocaleString()} samples`;
  } catch (error) {
    showError(error.message);
    const status = await request("/api/live").catch(() => null);
    if (status && !status.active) {
      state.liveView = false;
      state.liveGeneration = (state.liveGeneration || 0) + 1;
      if (livePoll) clearInterval(livePoll);
      if (spectrumPoll) clearTimeout(spectrumPoll);
      resetOutputSession();
      state.waterfall = null;
      state.view = null;
      state.tracks = [];
      state.regions = [];
      el("workspace").classList.add("hidden");
      el("empty-state").classList.remove("hidden");
      el("live-status").textContent = "Input stopped · review saved or retained audio";
    }
  }
  finally {
    if (!state.liveView) livePoll = null;
    el("live-start").disabled = state.liveView;
    el("live-stop").disabled = !state.liveView;
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
    const error = new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

el("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.liveView) { showError("Stop live capture before opening another source."); return; }
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
  if (state.liveView) { showError("Stop live capture before importing another source."); return; }
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
  const generation = state.captureGeneration = (state.captureGeneration || 0) + 1;
  clearDemodExports();
  state.liveView = false;
  resetOutputSession();
  state.waterfall = null;
  baseCanvas.getContext("2d").clearRect(0, 0, baseCanvas.width, baseCanvas.height);
  overlayCanvas.getContext("2d").clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  el("signal-labels")?.replaceChildren();
  state.view = null;
  state.selection = null;
  updateDecoderSelectionControls();
  updateModelButton();
  el("demod-run").disabled = false;
  el("ook-run").disabled = false;
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
  const spectrum = await request(`/api/captures/${c.id}/waterfall?fft_size=4096&hop_size=512&max_rows=640`);
  if (generation !== state.captureGeneration || state.liveView) return;
  state.waterfall = spectrum;
  drawWaterfall();
  const regions = await request(`/api/captures/${c.id}/regions`);
  if (generation !== state.captureGeneration || state.liveView) return;
  state.regions = regions;
  renderRegions();
  try {
    state.decodeRuns = await request(`/api/captures/${c.id}/decode-runs`);
  } catch (error) {
    state.decodeRuns = [];
    el("decode-runs").textContent = `Saved receive runs unavailable · ${error.message}`;
  }
  if (generation === state.captureGeneration && !state.liveView) renderDecodeRuns();
  try {
    const result = await request(`/api/captures/${c.id}/tracks`);
    if (generation !== state.captureGeneration || state.liveView) return;
    state.tracks = result.tracks;
    el("track-status").textContent = `Spectral heuristic · ${result.analysis.fft_size}-point FFT · candidates are provisional`;
  } catch (error) {
    if (generation !== state.captureGeneration || state.liveView) return;
    state.tracks = [];
    el("track-status").textContent = `Candidate analysis unavailable · ${error.message}`;
  }
  renderTracks();
  el("waterfall-state").textContent = `${state.waterfall.db.length} spectrum rows · ${state.waterfall.fft_size}-point FFT`;
}

function renderDecodeRuns() {
  const list = el("decode-runs");
  list.replaceChildren();
  const runs = state.decodeRuns || [];
  if (!runs.length) {
    list.textContent = "No saved Fldigi receive runs for this capture.";
    return;
  }
  for (const run of runs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "decode-run";
    const interval = Number.isInteger(run.sample_start) && Number.isInteger(run.sample_count) && run.sample_rate
      ? ` · ${(run.sample_start / run.sample_rate).toFixed(2)}–${((run.sample_start + run.sample_count) / run.sample_rate).toFixed(2)} s`
      : " · track time unavailable";
    button.textContent = `${run.current_mode || "Fldigi"} · ${run.status}${interval}`;
    button.addEventListener("click", () => {
      showOutput("text");
      state.textOutputContext = `Saved run ${run.id.slice(0,8)} · ${run.output_scope} · ${run.association}`;
      el("output-context").textContent = state.textOutputContext;
      el("decoded-text").textContent = `${run.status.toUpperCase()} · ${run.current_mode || "Fldigi"}\n${run.note}\n\n${run.output_text || "No decoded text was recorded."}`;
      el("decoded-text").scrollTop = 0;
    });
    list.append(button);
  }
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
  if (!state.view) resetWaterfallView(false);
  const view = waterfallViewBounds();
  const bins = view.frequencyEnd - view.frequencyStart;
  const rows = state.liveView ? Math.ceil(30 * data.sample_rate / data.hop_size) : view.rowEnd - view.rowStart;
  if (bins < 1 || rows < 1) return;
  baseCanvas.width = bins;
  baseCanvas.height = rows;
  overlayCanvas.width = bins;
  overlayCanvas.height = rows;
  const context = baseCanvas.getContext("2d", { alpha: false });
  const image = context.createImageData(bins, rows);
  const palette = state.palette ||= Array.from({length:256},(_,i)=>colorFor(-95+i*75/255,-95,-20));
  for (let p=0;p<image.data.length;p+=4) {
    image.data[p]=8; image.data[p+1]=17; image.data[p+2]=35; image.data[p+3]=255;
  }
  for (let index=view.rowStart;index<view.rowEnd;index++) {
    const y = state.liveView
      ? Math.floor((state.view.timeEndSample-data.sample_starts[index])/(state.view.timeEndSample-state.view.timeStartSample)*rows)
      : view.rowEnd - 1 - index;
    if (y<0 || y>=rows) continue;
    for (let x=0;x<bins;x++) {
      const level = Math.max(0,Math.min(255,Math.round((data.db[index][x+view.frequencyStart]+95)*255/75)));
      const [r,g,b] = palette[level];
      const p=(y*bins+x)*4;
      image.data[p]=r; image.data[p+1]=g; image.data[p+2]=b;
    }
  }
  context.putImageData(image, 0, 0);
  const firstFrequency = data.frequency_hz[view.frequencyStart];
  const lastFrequency = data.frequency_hz[view.frequencyEnd - 1];
  el("freq-low").textContent = `${Math.round(firstFrequency).toLocaleString()} Hz`;
  el("freq-high").textContent = `${Math.round(lastFrequency).toLocaleString()} Hz`;
  const firstTime = data.sample_starts[view.rowStart] / data.sample_rate;
  const lastTime = Math.min(data.sample_count, data.sample_starts[view.rowEnd - 1] + data.hop_size) / data.sample_rate;
  if (state.liveView) {
    const labels = liveTimeLabels(Math.max(0, state.view.timeStartSample), state.view.timeEndSample, data.sample_rate);
    el("time-start").textContent = labels.end;
    el("time-end").textContent = "−30 s";
  } else {
    el("time-start").textContent = `${lastTime.toFixed(2)} s`;
    el("time-end").textContent = `${firstTime.toFixed(2)} s`;
  }
  drawOverlays();
}

function lowerBound(values, value) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (values[middle] < value) low = middle + 1;
    else high = middle;
  }
  return low;
}

function waterfallViewBounds() {
  const data = state.waterfall;
  const frequencyStart = Math.max(0, Math.min(data.frequency_hz.length - 1, lowerBound(data.frequency_hz, state.view.frequencyLowHz)));
  const frequencyEnd = Math.max(frequencyStart + 1, Math.min(data.frequency_hz.length, lowerBound(data.frequency_hz, state.view.frequencyHighHz) + 1));
  const rowStart = Math.max(0, Math.min(data.sample_starts.length - 1, lowerBound(data.sample_starts, state.view.timeStartSample)));
  const rowEnd = Math.max(rowStart + 1, Math.min(data.sample_starts.length, lowerBound(data.sample_starts, state.view.timeEndSample)));
  return { frequencyStart, frequencyEnd, rowStart, rowEnd };
}

function resetWaterfallView(redraw = true) {
  const data = state.waterfall;
  if (!data) return;
  const timeEndSample = data.sample_count;
  state.view = {
    frequencyLowHz: 0,
    frequencyHighHz: Math.min(3000, data.sample_rate / 2),
    timeStartSample: state.liveView ? timeEndSample - 30 * data.sample_rate : 0,
    timeEndSample,
    followTime: state.liveView,
  };
  reprojectSelection();
  if (redraw) drawWaterfall();
}

function reprojectSelection() {
  if (!state.selection || !state.waterfall || !state.view) return;
  const data = state.waterfall;
  const view = waterfallViewBounds();
  const lowHz = data.frequency_hz[view.frequencyStart];
  const highHz = data.frequency_hz[view.frequencyEnd - 1];
  const bins = view.frequencyEnd - view.frequencyStart;
  const x = (frequency) => Math.round((frequency - lowHz) / Math.max(1, highHz - lowHz) * (bins - 1));
  state.selection.x1 = x(state.selection.frequency_low_hz);
  state.selection.x2 = x(state.selection.frequency_high_hz);
  state.selection.y1 = sampleToRow(state.selection.sample_start);
  state.selection.y2 = sampleToRow(state.selection.sample_start + state.selection.sample_count);
}

function sampleToRow(sample) {
  if (state.liveView && state.view) {
    return (state.view.timeEndSample-sample)/(state.view.timeEndSample-state.view.timeStartSample)*overlayCanvas.height;
  }
  const starts = state.waterfall.sample_starts;
  let lo = 0;
  let hi = starts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (starts[mid] < sample) lo = mid + 1;
    else hi = mid;
  }
  const bounds = waterfallViewBounds();
  return Math.max(0, Math.min(bounds.rowEnd - bounds.rowStart - 1, bounds.rowEnd - 1 - lo));
}

function drawOverlays() {
  const data = state.waterfall;
  if (!data) return;
  if (!state.view) resetWaterfallView(false);
  const view = waterfallViewBounds();
  const ctx = overlayCanvas.getContext("2d");
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const lowHz = data.frequency_hz[view.frequencyStart];
  const highHz = data.frequency_hz[view.frequencyEnd - 1];
  const bins = view.frequencyEnd - view.frequencyStart;
  const visibleRow = (sample) => sampleToRow(sample);
  const visibleX = (frequency) => Math.round((frequency - lowHz) / Math.max(1, highHz - lowHz) * (bins - 1));
  for (const track of state.tracks) {
    const rowA = visibleRow(track.sample_start);
    const rowB = visibleRow(track.sample_start + track.sample_count);
    const x = visibleX(track.frequency_low_hz);
    const right = visibleX(track.frequency_high_hz);
    if (rowB < 0 || rowA >= overlayCanvas.height || right < 0 || x >= bins) continue;
    ctx.strokeStyle = track.track_state === "persistent" ? "#52d6d0" : "#f1be69";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    const left = Math.max(0, x);
    const top = Math.max(0, rowA);
    ctx.strokeRect(left, top, Math.max(2, Math.min(bins - 1, right) - left), Math.max(2, Math.min(overlayCanvas.height - 1, rowB) - top));
  }
  ctx.setLineDash([]);
  for (const region of state.regions) {
    const rowA = visibleRow(region.sample_start);
    const rowB = visibleRow(region.sample_start + region.sample_count);
    const low = region.frequency_low_hz ?? 0;
    const high = region.frequency_high_hz ?? data.sample_rate / 2;
    const x = visibleX(low);
    const right = visibleX(high);
    if (rowB < 0 || rowA >= overlayCanvas.height || right < 0 || x >= bins) continue;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    const left = Math.max(0, x);
    const top = Math.max(0, rowA);
    ctx.strokeRect(left, top, Math.max(2, Math.min(bins - 1, right) - left), Math.max(2, Math.min(overlayCanvas.height - 1, rowB) - top));
  }
  if (state.selection) {
    const { x1, y1, x2, y2 } = state.selection;
    ctx.strokeStyle = "#72fff0";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.max(2, Math.abs(x2 - x1)), Math.max(2, Math.abs(y2 - y1)));
    ctx.setLineDash([]);
  }
  renderSignalLabels();
}

function selectTrack(track) {
  const data = state.waterfall;
  if (!data) return;
  if (!state.view) resetWaterfallView(false);
  const view = waterfallViewBounds();
  const lowHz = data.frequency_hz[view.frequencyStart];
  const highHz = data.frequency_hz[view.frequencyEnd - 1];
  const bins = view.frequencyEnd - view.frequencyStart;
  const x1 = Math.max(0, Math.min(bins - 1, Math.round((track.frequency_low_hz - lowHz) / Math.max(1, highHz - lowHz) * (bins - 1))));
  const x2 = Math.max(x1 + 1, Math.min(bins - 1, Math.round((track.frequency_high_hz - lowHz) / Math.max(1, highHz - lowHz) * (bins - 1))));
  const y1 = Math.max(0, sampleToRow(track.sample_start));
  const y2 = Math.max(y1 + 1, sampleToRow(track.sample_start + track.sample_count));
  const binStep = (highHz - lowHz) / Math.max(1, bins - 1);
  const sampleEnd = Math.min(data.sample_count, track.sample_start + track.sample_count);
  state.selection = { sample_start: track.sample_start, sample_count: Math.max(1, sampleEnd - track.sample_start), frequency_low_hz: track.frequency_low_hz, frequency_high_hz: track.frequency_high_hz, x1, y1, x2, y2 };
  state.selectedTrack = track;
  beginNewRegion();
  const center = (track.frequency_low_hz + track.frequency_high_hz) / 2;
  const bandwidth = track.frequency_high_hz - track.frequency_low_hz;
  el("inspector-context").textContent = `${track.modulation_hypothesis} candidate · ${center.toFixed(0)} Hz center · ${bandwidth.toFixed(0)} Hz wide · peak SNR estimate ${track.peak_snr_db?.toFixed(1) ?? "unknown"} dB · ${track.tone_count_candidate ?? "unknown"} tone centers${Number.isFinite(track.drift_hz_per_second) ? ` · drift ${track.drift_hz_per_second.toFixed(1)} Hz/s` : ""}`;
  openInspector("label");
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
}

function applyTrackToneEstimate(track) {
  selectTrack(track);
  el("mfsk-count").value = String(track.tone_count_candidate);
  el("mfsk-base").value = String(track.tone_centers_hz[0]);
  el("mfsk-spacing").value = String(track.tone_spacing_hz_candidate);
  el("mfsk-auto-timing").checked = true;
  el("mfsk-auto-rate").checked = true;
  el("mfsk-output").textContent = `Applied provisional tone bank: ${track.tone_centers_hz.map((tone) => tone.toFixed(1)).join(", ")} Hz · spacing ${track.tone_spacing_hz_candidate.toFixed(1)} Hz · grid residual ${track.tone_grid_residual_hz.toFixed(1)} Hz. Review these estimates, then run M-FSK analysis.`;
  openInspector("raw");
  showRawTool("mfsk");
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
      const drift = Number.isFinite(track.drift_hz_per_second) ? track.drift_hz_per_second : 0;
      const identity = track.track_id ? `${track.track_id.split(":").at(-1)} · ${track.track_state}` : "single analysis";
      const proposal = track.model_proposal;
      const modelLabel = proposal?.available
        ? proposal.noise_detected
          ? ` · model noise-only (${(100 * proposal.confidence).toFixed(0)}%)`
          : proposal.novelty_rejected
          ? ` · model abstained (outside trained range)`
          : proposal.abstained
          ? ` · model abstained (${proposal.top_candidate}, ${(100 * proposal.confidence).toFixed(0)}%)`
          : ` · model ${proposal.method_id} ${(100 * proposal.confidence).toFixed(0)}%`
        : "";
      summary.textContent = `${identity} · ${track.modulation_hypothesis} · ${track.tone_count_candidate} tone centers${modelLabel}`;
      const detail = document.createElement("div");
      detail.className = "region-detail";
      detail.textContent = `${track.frequency_low_hz.toFixed(0)}–${track.frequency_high_hz.toFixed(0)} Hz · ${track.time_start_seconds.toFixed(2)}–${track.time_end_seconds.toFixed(2)} s · ${track.peak_snr_db.toFixed(1)} dB peak SNR · drift ${drift >= 0 ? "+" : ""}${drift.toFixed(1)} Hz/s`;
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Review region";
      button.addEventListener("click", () => selectTrack(track));
      row.append(summary, detail, button);
      if (track.tone_grid_regular && track.tone_count_candidate >= 2 && track.tone_count_candidate <= 64) {
        const toneButton = document.createElement("button");
        toneButton.type = "button";
        toneButton.textContent = "Use tone estimates";
        toneButton.addEventListener("click", () => applyTrackToneEstimate(track));
        row.append(toneButton);
      }
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
  const view = waterfallViewBounds();
  const rect = overlayCanvas.getBoundingClientRect();
  const moved = Math.hypot((state.drag.x - point.x) * rect.width / overlayCanvas.width, (state.drag.y - point.y) * rect.height / overlayCanvas.height);
  if (moved < 5) {
    const sample = sampleAtCanvasRow(point.y);
    const frequency = data.frequency_hz[Math.min(view.frequencyEnd - 1, view.frequencyStart + point.x)];
    const track = hitSignal(state.tracks, sample, frequency);
    if (track) { state.drag = null; selectTrack(track); return; }
  }
  const rowStart = Math.min(state.drag.y, point.y);
  const rowEnd = Math.max(state.drag.y, point.y);
  const firstSample = sampleAtCanvasRow(rowStart);
  const lastSample = sampleAtCanvasRow(rowEnd);
  const sampleStart = Math.max(0, Math.floor(Math.min(firstSample, lastSample)));
  const sampleEnd = state.liveView
    ? Math.min(data.sample_count, Math.ceil(Math.max(firstSample, lastSample)))
    : Math.min(data.sample_count, Math.ceil(Math.max(firstSample, lastSample)) + data.fft_size);
  if (state.liveView && sampleEnd <= sampleStart) {
    state.drag = null;
    state.selection = null;
    drawOverlays();
    return;
  }
  const firstBin = view.frequencyStart;
  const xLow = Math.min(state.drag.x, point.x);
  const xHigh = Math.max(state.drag.x, point.x);
  const frequencyLow = data.frequency_hz[Math.min(view.frequencyEnd - 1, firstBin + xLow)];
  const frequencyHigh = data.frequency_hz[Math.min(view.frequencyEnd - 1, firstBin + Math.max(xLow + 1, xHigh))];
  state.selection = {
    sample_start: sampleStart,
    sample_count: Math.max(1, sampleEnd - sampleStart),
    frequency_low_hz: frequencyLow,
    frequency_high_hz: frequencyHigh,
    x1: state.drag.x, y1: state.drag.y, x2: point.x, y2: point.y,
  };
  state.drag = null;
  state.selectedTrack = null;
  beginNewRegion();
  openInspector("label");
  el("selection-summary").textContent = `${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz`;
  const t0 = sampleStart / data.sample_rate;
  const t1 = sampleEnd / data.sample_rate;
  el("selection-details").textContent = `Samples ${sampleStart.toLocaleString()}–${sampleEnd.toLocaleString()} · ${t0.toFixed(3)}–${t1.toFixed(3)} s`;
  el("inspector-context").textContent = `Manual region · ${frequencyLow.toFixed(0)}–${frequencyHigh.toFixed(0)} Hz · ${t0.toFixed(2)}–${t1.toFixed(2)} s${state.liveView ? " · stop recording to save labels" : ""}`;
  el("save-region").disabled = state.liveView || !state.capture;
  updateModelButton();
  drawOverlays();
});

el("save-region").addEventListener("click", async () => {
  if (!state.selection || !state.capture) return;
  clearError();
  const button = el("save-region");
  const captureId = state.capture.id;
  const edit = state.editRegion;
  button.disabled = true;
  try {
    const selection = state.selection;
    const destination = edit ? `/api/captures/${captureId}/regions/${edit.id}/labels` : `/api/captures/${captureId}/regions`;
    await request(destination, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...(edit ? {expected_revision:edit.revision} : {}),
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
    const regions = await request(`/api/captures/${captureId}/regions`);
    if (state.capture?.id !== captureId) return;
    state.regions = regions;
    beginNewRegion();
    renderRegions();
    state.selection = null;
    updateModelButton();
    el("selection-summary").textContent = "No region selected";
    el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
    for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = "";
    drawOverlays();
  } catch (error) {
    if (state.capture?.id !== captureId) return;
    showError(error.message);
    if (error.status === 409) {
      state.regions = await request(`/api/captures/${captureId}/regions`);
      renderRegions();
      el("label-state").textContent = "A newer label exists. Reopen its marker to review it; your draft is retained here.";
    }
    button.disabled = false;
  }
});

el("demod-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  const sourceCapture = state.capture;
  clearError();
  const button = el("demod-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("demod-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const settings = {
      sample_start: selection?.sample_start ?? 0,
      sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
      symbol_rate: Number(el("demod-rate").value),
      tone_0_hz: Number(el("demod-tone0").value),
      tone_1_hz: Number(el("demod-tone1").value),
      timing_offset_samples: Number(el("demod-offset").value),
      auto_timing: el("demod-auto-timing").checked,
    };
    const result = await request(`/api/captures/${sourceCapture.id}/demod/2fsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (state.capture !== sourceCapture) return;
    storeDemodExport("2fsk", result, settings);
    const confidence = result.soft_metrics.length
      ? result.soft_metrics.reduce((sum, value) => sum + Math.abs(value), 0) / result.soft_metrics.length
      : 0;
    const groupedBits = result.bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    const timing = result.timing_offset_estimated
      ? ` · estimated timing ${result.timing_offset_samples.toFixed(2)} samples (${result.timing_search.phase_count} phases)`
      : ` · timing ${result.timing_offset_samples.toFixed(2)} samples`;
    el("demod-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols${timing} · mean |soft metric| ${confidence.toFixed(3)}\nProtocol: unknown\n\n${groupedBits}`;
  } catch (error) {
    if (state.capture !== sourceCapture) return;
    el("demod-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover raw bits";
  }
});

el("ook-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  const sourceCapture = state.capture;
  clearError();
  const button = el("ook-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("ook-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const settings = {
      sample_start: selection?.sample_start ?? 0,
      sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
      symbol_rate: Number(el("ook-rate").value),
      carrier_hz: Number(el("ook-carrier").value),
      timing_offset_samples: Number(el("ook-offset").value),
      auto_timing: el("ook-auto-timing").checked,
    };
    const result = await request(`/api/captures/${sourceCapture.id}/demod/ook`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (state.capture !== sourceCapture) return;
    storeDemodExport("ook", result, settings);
    const groupedBits = result.bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    const timing = result.timing_offset_estimated
      ? `estimated ${result.timing_offset_samples.toFixed(2)} samples (${result.timing_search.phase_count} phases)`
      : `${result.timing_offset_samples.toFixed(2)} samples`;
    el("ook-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols · timing ${timing}\nEstimated amplitude 0/1: ${result.estimated_zero_amplitude.toFixed(4)} / ${result.estimated_one_amplitude.toFixed(4)} · level separation ${result.level_separation_score.toFixed(3)}\nProtocol: unknown\n\n${groupedBits}`;
  } catch (error) {
    if (state.capture !== sourceCapture) return;
    el("ook-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover OOK bits";
  }
});

el("mfsk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  const sourceCapture = state.capture;
  clearError();
  const button = el("mfsk-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("mfsk-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const settings = {
      sample_start: selection?.sample_start ?? 0,
      sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
      symbol_rate: Number(el("mfsk-rate").value),
      tone_count: Number(el("mfsk-count").value),
      base_tone_hz: Number(el("mfsk-base").value),
      tone_spacing_hz: Number(el("mfsk-spacing").value),
      timing_offset_samples: Number(el("mfsk-offset").value),
      auto_timing: el("mfsk-auto-timing").checked,
      auto_rate_search: el("mfsk-auto-rate").checked,
      rate_search_min: Number(el("mfsk-rate-min").value),
      rate_search_max: Number(el("mfsk-rate-max").value),
      rate_search_steps: Number(el("mfsk-rate-steps").value),
    };
    const result = await request(`/api/captures/${sourceCapture.id}/demod/mfsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (state.capture !== sourceCapture) return;
    storeDemodExport("mfsk", result, settings);
    const confidence = result.soft_confidence.length
      ? result.soft_confidence.reduce((sum, value) => sum + value, 0) / result.soft_confidence.length
      : 0;
    const groupedSymbols = result.symbols.join(" ");
    const bits = result.bits === null ? "Bits: unavailable for non-power-of-two tone count" : `Bits:\n${result.bits.match(/.{1,64}/g)?.join("\n") || ""}`;
    const rate = result.symbol_rate_estimated
      ? `estimated ${result.symbol_rate.toFixed(2)} baud (${result.rate_search.candidate_rates.length} hypotheses)`
      : `${result.symbol_rate.toFixed(2)} baud`;
    const bestRates = result.rate_search
      ? `\nBest rate hypotheses: ${result.rate_search.candidate_rates
        .map((candidate, index) => ({ candidate, score: result.rate_search.candidate_scores[index] }))
        .sort((left, right) => right.score - left.score)
        .slice(0, 3)
        .map(({ candidate, score }) => `${candidate.toFixed(2)} baud (${score.toFixed(3)})`)
        .join(", ")}`
      : "";
    const timing = result.timing_offset_estimated
      ? ` · estimated timing ${result.timing_offset_samples.toFixed(2)} samples (${result.timing_search.phase_count} phases)`
      : ` · timing ${result.timing_offset_samples.toFixed(2)} samples`;
    el("mfsk-output").textContent = `${result.demodulator} · ${result.tone_count} tones · ${result.symbol_count} symbols · ${rate}${timing} · mean soft confidence ${confidence.toFixed(3)}${bestRates}\nProtocol: unknown\n\nTone indices:\n${groupedSymbols}\n\n${bits}`;
  } catch (error) {
    if (state.capture !== sourceCapture) return;
    el("mfsk-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover tone symbols";
  }
});

el("bpsk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  const sourceCapture = state.capture;
  clearError();
  const button = el("bpsk-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("bpsk-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const settings = {
      sample_start: selection?.sample_start ?? 0,
      sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
      symbol_rate: Number(el("bpsk-rate").value),
      carrier_hz: Number(el("bpsk-carrier").value),
      timing_offset_samples: Number(el("bpsk-offset").value),
    };
    const result = await request(`/api/captures/${sourceCapture.id}/demod/bpsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (state.capture !== sourceCapture) return;
    storeDemodExport("bpsk", result, settings);
    const grouped = (bits) => bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    const confidence = result.soft_metrics.length
      ? result.soft_metrics.reduce((sum, value) => sum + Math.abs(value), 0) / result.soft_metrics.length
      : 0;
    el("bpsk-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols · mean |soft metric| ${confidence.toFixed(3)}\nPolarity ambiguous: ${result.polarity_ambiguous}\nProtocol: unknown\n\nBit hypothesis A:\n${grouped(result.bits)}\n\nBit hypothesis B (inverted):\n${grouped(result.inverted_bits)}`;
  } catch (error) {
    if (state.capture !== sourceCapture) return;
    el("bpsk-output").textContent = `Demodulation failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView;
    button.textContent = "Recover BPSK bits";
  }
});

el("qpsk-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.capture || state.liveView) return;
  const sourceCapture = state.capture;
  clearError();
  const button = el("qpsk-run");
  button.disabled = true;
  button.textContent = "Recovering…";
  el("qpsk-output").textContent = "Analyzing selected audio…";
  try {
    const selection = state.selection;
    const settings = {
      sample_start: selection?.sample_start ?? 0,
      sample_count: Math.min(1_000_000, selection?.sample_count ?? Math.min(120_000, state.capture.sample_count)),
      symbol_rate: Number(el("qpsk-rate").value),
      carrier_hz: Number(el("qpsk-carrier").value),
      timing_offset_samples: Number(el("qpsk-offset").value),
    };
    const result = await request(`/api/captures/${sourceCapture.id}/demod/qpsk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(settings),
    });
    if (state.capture !== sourceCapture) return;
    storeDemodExport("qpsk", result, settings);
    const grouped = (bits) => bits.match(/.{1,64}/g)?.join("\n") || "(no complete symbols)";
    const hypotheses = result.symbol_hypotheses.map((item, index) =>
      `Hypothesis ${index + 1} · rotation ${(item.rotation_radians * 180 / Math.PI).toFixed(0)}°\nSymbols: ${item.symbols.join(" ")}\nBits:\n${grouped(item.bits)}`
    ).join("\n\n");
    el("qpsk-output").textContent = `${result.demodulator} · ${result.symbol_count} symbols · absolute rotation unknown\nProtocol: unknown\n\n${hypotheses}`;
  } catch (error) {
    if (state.capture !== sourceCapture) return;
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
      const review = document.createElement("button");
      review.type = "button";
      review.textContent = "Review";
      review.addEventListener("click", () => selectSavedRegion(region));
      row.append(review);
      list.append(row);
    }
  }
  renderRegionShortcuts();
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

// Workspace navigation keeps capture and selection state in one place.
function openInspector(tab = "label") {
  el("inspector").classList.remove("hidden");
  for (const node of document.querySelectorAll("[data-inspector-view]")) node.hidden = node.dataset.inspectorView !== tab;
  for (const button of document.querySelectorAll("[data-inspector-tab]")) button.setAttribute("aria-pressed", String(button.dataset.inspectorTab === tab));
  updateDecoderSelectionControls();
}
function updateDecoderSelectionControls() {
  const validSelection = Boolean(state.capture && state.selection && !state.liveView);
  const validMode = Boolean(el("fldigi-mode")?.value && !el("fldigi-mode").disabled);
  if (el("decode-selection")) el("decode-selection").disabled = !validSelection || !validMode;
  if (el("tag-as-mode")) el("tag-as-mode").disabled = !validSelection || !validMode;
}
function showOutput(tab) {
  state.outputTab = tab;
  for (const node of document.querySelectorAll("[data-output-view]")) node.hidden = node.dataset.outputView !== tab;
  for (const button of document.querySelectorAll("[data-output-tab]")) button.setAttribute("aria-pressed", String(button.dataset.outputTab === tab));
  el("output-dock").classList.remove("collapsed");
  el("toggle-output").textContent = "Collapse";
  el("toggle-output").setAttribute("aria-expanded", "true");
  updateOutputContext();
}
function showRawTool(mode) {
  el("raw-tool").value = mode;
  for (const node of document.querySelectorAll("[data-raw-tool]")) node.hidden = node.dataset.rawTool !== mode;
  for (const node of document.querySelectorAll("[data-raw-result]")) node.hidden = node.dataset.rawResult !== mode;
  updateOutputContext();
}
function resetOutputSession() {
  state.outputGeneration = (state.outputGeneration || 0) + 1;
  fldigiOffset = null;
  state.textOutputContext = "Shared Fldigi output · audio route external";
  el("decoded-text").textContent = "Decoded text will appear here.";
  for (const id of ["demod-output", "mfsk-output", "ook-output", "bpsk-output", "qpsk-output"]) {
    el(id).textContent = "Select a saved capture or region to analyze.";
  }
  showOutput("text");
  el("output-context").textContent = "Shared Fldigi output · audio route external";
  if (trialPoll) clearInterval(trialPoll);
  trialPoll = null;
  watchedTrialTrack = null;
  state.selectedTrack = null;
  state.frozen = false;
  state.decoderTrial = null;
  el("signal-region-shortcuts").replaceChildren();
  beginNewRegion();
  el("inspector-context").textContent = "Select a signal on the waterfall to inspect it.";
  el("manifest-download").classList.add("hidden");
  el("audio-player").hidden = state.liveView;
  el("freeze-view").textContent = "Freeze view";
  el("freeze-view").disabled = !state.liveView;
  for (const dialog of document.querySelectorAll("dialog[open]")) dialog.close();
  el("inspector").classList.add("hidden");
}
function renderSignalLabels() {
  let container = el("signal-labels");
  if (!container) {
    container = document.createElement("div");
    container.id = "signal-labels";
    container.className = "signal-labels";
    overlayCanvas.parentElement.append(container);
  }
  const data = state.waterfall;
  if (!data?.db.length) { container.replaceChildren(); return; }
  const bounds = waterfallViewBounds();
  const width = container.clientWidth;
  const height = container.clientHeight;
  if (!width || !height) return;
  const low = data.frequency_hz[bounds.frequencyStart];
  const high = data.frequency_hz[bounds.frequencyEnd - 1];
  const items = [];
  for (const [index, track] of state.tracks.entries()) {
    const start = Math.min(sampleToRow(track.sample_start), sampleToRow(track.sample_start + track.sample_count));
    const end = Math.max(sampleToRow(track.sample_start), sampleToRow(track.sample_start + track.sample_count));
    if (end < 0 || start >= overlayCanvas.height || track.frequency_high_hz < low || track.frequency_low_hz > high) continue;
    const trial = state.decoderTrial?.track_id === track.track_id ? state.decoderTrial : null;
    const activity = trial?.status === "running" ? ` · trying ${trial.mode}` : trial?.status === "tentative_output" ? ` · tentative ${trial.mode} output` : "";
    items.push({
      track, text: signalCaption(track, index) + activity,
      selected: state.selectedTrack === track || Boolean(track.track_id && state.selectedTrack?.track_id === track.track_id),
      x: (track.frequency_low_hz - low) / Math.max(1, high - low) * width,
      y: Math.max(0, start / overlayCanvas.height * height),
    });
  }
  for (const region of state.regions) {
    const start = Math.min(sampleToRow(region.sample_start), sampleToRow(region.sample_start + region.sample_count));
    const end = Math.max(sampleToRow(region.sample_start), sampleToRow(region.sample_start + region.sample_count));
    if (end < 0 || start >= overlayCanvas.height || (region.frequency_high_hz ?? high) < low || (region.frequency_low_hz ?? low) > high) continue;
    items.push({region, text: `Saved · ${region.variant || region.protocol || region.method || region.family || "Unknown"}`,
      x: ((region.frequency_low_hz ?? low) - low) / Math.max(1, high - low) * width,
      y: Math.max(0, start / overlayCanvas.height * height)});
  }
  const layout = layoutSignalLabels(items, width, height);
  // Preserve focused controls when possible during live refresh.
  const focused = document.activeElement?.dataset.signalKey;
  container.replaceChildren();
  const leaders = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  leaders.setAttribute("viewBox", `0 0 ${width} ${height}`);
  leaders.setAttribute("class", "signal-leaders");
  leaders.setAttribute("aria-hidden", "true");
  container.append(leaders);
  for (const item of layout.placed) {
    if (item.y !== item.anchorY || item.x !== item.anchorX) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", String(Math.max(0, Math.min(width,item.anchorX))));
      line.setAttribute("y1", String(item.anchorY));
      line.setAttribute("x2", String(item.x+6));
      line.setAttribute("y2", String(item.y+12));
      leaders.append(line);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `signal-tag ${item.region ? "saved" : "candidate"}${item.selected ? " selected" : ""}`;
    button.textContent = item.text;
    button.style.left = `${item.x}px`;
    button.style.top = `${item.y}px`;
    button.style.width = `${item.width}px`;
    button.dataset.signalKey = item.track?.track_id || item.region?.id || item.text;
    button.title = item.track
      ? `${item.text}\n${item.track.frequency_low_hz.toFixed(0)}–${item.track.frequency_high_hz.toFixed(0)} Hz · peak SNR estimate ${item.track.peak_snr_db?.toFixed(1) ?? "unknown"} dB\n${item.track.tone_count_candidate ?? "Unknown"} tone centers${Number.isFinite(item.track.tone_spacing_hz_candidate) ? ` · spacing ${item.track.tone_spacing_hz_candidate.toFixed(1)} Hz` : ""}${Number.isFinite(item.track.drift_hz_per_second) ? `\nDrift ${item.track.drift_hz_per_second.toFixed(1)} Hz/s` : ""}\nClick to inspect; identification remains provisional.`
      : `${item.text} · ${item.region.source} / ${item.region.review_state}`;
    const showDetail = () => {
      el("signal-hover-detail")?.remove();
      const detail = document.createElement("div");
      detail.id = "signal-hover-detail";
      detail.className = "signal-tooltip";
      detail.setAttribute("role", "tooltip");
      detail.textContent = button.title;
      container.append(detail);
      detail.style.left = `${Math.max(0,Math.min(item.x,width-detail.offsetWidth))}px`;
      detail.style.top = `${Math.max(0,Math.min(item.y+28,height-detail.offsetHeight))}px`;
      button.setAttribute("aria-describedby", detail.id);
    };
    button.addEventListener("mouseenter", showDetail);
    button.addEventListener("focus", showDetail);
    for (const event of ["mouseleave", "blur"]) button.addEventListener(event, () => el("signal-hover-detail")?.remove());
    button.addEventListener("click", () => item.track ? selectTrack(item.track) : selectSavedRegion(item.region));
    container.append(button);
    if (button.dataset.signalKey === focused) button.focus({preventScroll:true});
  }
  if (layout.overflow.length) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "signal-overflow";
    button.textContent = `+${layout.overflow.length} signals / labels`;
    button.addEventListener("click", () => openInspector("signals"));
    container.append(button);
  }
}
function selectSavedRegion(region) {
  const data = state.waterfall;
  if (!data) return;
  clearError();
  state.selection = {sample_start:region.sample_start, sample_count:region.sample_count,
    frequency_low_hz:region.frequency_low_hz ?? 0, frequency_high_hz:region.frequency_high_hz ?? Math.min(3000,data.sample_rate/2)};
  state.selectedTrack = null;
  state.editRegion = {id:region.id, revision:region.revision};
  reprojectSelection();
  for (const field of ["family","method","protocol","variant","notes"]) el(field).value = region[field] || "";
  el("selection-summary").textContent = region.variant || region.protocol || region.method || region.family || "Saved unknown region";
  el("selection-details").textContent = `${region.source} · ${region.review_state} · ${(region.sample_start/data.sample_rate).toFixed(2)}–${((region.sample_start+region.sample_count)/data.sample_rate).toFixed(2)} s`;
  el("label-state").textContent = `Editing label revision ${region.revision} · original history retained`;
  el("save-region").textContent = "Save revision";
  el("inspector-context").textContent = `Saved annotation · ${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz · revision ${region.revision}`;
  loadLabelHistory(region, state.capture.id);
  el("save-region").disabled = state.liveView;
  updateModelButton();
  for (const dialog of document.querySelectorAll("dialog[open]")) dialog.close();
  openInspector("label");
  drawOverlays();
}
async function refreshLibrary() {
  const target = el("capture-list");
  try {
    const captures = await request("/api/captures");
    target.replaceChildren();
    for (const capture of captures) {
      const row = document.createElement("div");
      row.className = "capture-row";
      const title = document.createElement("span");
      title.textContent = `${capture.source_metadata.uploaded_filename || capture.source_metadata.generator || capture.source_metadata.device_name || "Recording"} · ${(capture.sample_count / capture.sample_rate).toFixed(1)} s`;
      const open = document.createElement("button"); open.type = "button"; open.textContent = "Open";
      open.disabled = state.liveView;
      open.title = state.liveView ? "Stop live recording before opening another source" : "Open recording";
      open.addEventListener("click", async () => {
        try { state.capture = capture; await loadCapture(); } catch (error) { showError(error.message); }
      });
      row.append(title, open); target.append(row);
    }
    if (!captures.length) target.textContent = "No saved recordings yet.";
  } catch (error) { target.textContent = error.message; }
}
for (const button of document.querySelectorAll("[data-inspector-tab]")) button.addEventListener("click", () => openInspector(button.dataset.inspectorTab));
for (const button of document.querySelectorAll("[data-output-tab]")) button.addEventListener("click", () => showOutput(button.dataset.outputTab));
for (const button of document.querySelectorAll("[data-close-dialog]")) button.addEventListener("click", () => button.closest("dialog").close());
el("show-signals").addEventListener("click", () => openInspector("signals"));
el("close-inspector").addEventListener("click", () => el("inspector").classList.add("hidden"));
el("show-settings").addEventListener("click", () => el("settings-dialog").showModal());
for (const id of ["show-library", "open-recording"]) el(id).addEventListener("click", () => { refreshLibrary(); el("library-dialog").showModal(); });
el("raw-tool").addEventListener("change", () => showRawTool(el("raw-tool").value));
el("toggle-output").addEventListener("click", () => {
  const collapsed = el("output-dock").classList.toggle("collapsed");
  el("toggle-output").textContent = collapsed ? "Expand" : "Collapse";
  el("toggle-output").setAttribute("aria-expanded", String(!collapsed));
});
el("freeze-view").addEventListener("click", () => {
  state.frozen = !state.frozen;
  el("freeze-view").textContent = state.frozen ? "Resume view" : "Freeze view";
  el("live-status").textContent = state.frozen ? "Recording · view frozen" : "Recording · following live audio";
  if (!state.frozen) applyLiveDisplay();
});
document.addEventListener("keydown", event => { if (event.key === "Escape") el("inspector").classList.add("hidden"); });
overlayCanvas.addEventListener("pointermove", event => {
  if (!state.waterfall) return;
  const point = canvasPoint(event), bounds = waterfallViewBounds(), data = state.waterfall;
  const sample = sampleAtCanvasRow(point.y);
  const row = Math.min(bounds.rowEnd-1, lowerBound(data.sample_starts, sample));
  const bin = Math.min(bounds.frequencyEnd - 1, bounds.frequencyStart + point.x);
  if (sample < data.sample_starts[0] || Math.abs(data.sample_starts[row] - sample) > data.hop_size) {
    el("cursor-readout").textContent = `${data.frequency_hz[bin].toFixed(1)} Hz · no display samples`;
    return;
  }
  el("cursor-readout").textContent = `${data.frequency_hz[bin].toFixed(1)} Hz · ${(data.sample_starts[row]/data.sample_rate).toFixed(2)} s · ${data.db[row][bin].toFixed(1)} dB`;
});
// Move each analysis result into the dock, preserving API handlers and exports.
for (const mode of Object.keys(DEMOD_EXPORT_BUTTONS)) {
  const outputId = mode === "2fsk" ? "demod-output" : `${mode}-output`;
  const wrapper = document.createElement("div"); wrapper.dataset.rawResult = mode;
  wrapper.append(el(outputId), el(DEMOD_EXPORT_BUTTONS[mode]));
  el("raw-results").append(wrapper);
  const form = el(mode === "2fsk" ? "demod-form" : `${mode}-form`);
  form.addEventListener("submit", () => { showRawTool(mode); showOutput("raw"); });
}
el("raw-results").querySelector(":scope > p")?.remove();
showRawTool("2fsk");
showOutput("text");
new ResizeObserver(() => { if (state.waterfall) renderSignalLabels(); }).observe(overlayCanvas.parentElement);

function sampleAtCanvasRow(y) {
  if (state.liveView) return state.view.timeEndSample - y / Math.max(1,overlayCanvas.height) * (state.view.timeEndSample-state.view.timeStartSample);
  const view=waterfallViewBounds();
  return state.waterfall.sample_starts[Math.max(view.rowStart, Math.min(view.rowEnd - 1, view.rowEnd - 1 - Math.round(y)))];
}
function applyLiveDisplay() {
  if (!state.liveView || state.frozen || !liveDisplay?.data?.db.length) return;
  state.waterfall=liveDisplay.data;
  if (!state.view) resetWaterfallView(false);
  const window=liveDisplayBounds(state.waterfall.sample_count,state.waterfall.sample_rate);
  state.view.timeStartSample=window.startSample;
  state.view.timeEndSample=window.endSample;
  reprojectSelection();
  drawWaterfall();
  el("waterfall-state").textContent=`30 s display · ${state.waterfall.gap_count} update gaps · 10 s analysis`;
}
function startSpectrumStream(generation,sessionId) {
  if (spectrumPoll) clearTimeout(spectrumPoll);
  const update=async()=>{
    let delay=200;
    try {
      const after=liveDisplay?.lastSample ?? -1;
      const response=await fetch(`/api/live/spectrum?after_sample=${after}&session_id=${encodeURIComponent(sessionId)}`, {signal:AbortSignal.timeout(5000)});
      if (!response.ok) throw new Error(`Spectrum request ${response.status}`);
      const packet=decodeSpectrumPacket(await response.arrayBuffer());
      if (generation!==state.liveGeneration || !state.liveView) return;
      if (packet.session_id!==sessionId) throw new Error("Live source changed; stop and reopen capture");
      liveDisplay.append(packet);
      applyLiveDisplay();
    } catch(error) {
      delay=1000;
      if (generation===state.liveGeneration && state.liveView) el("waterfall-state").textContent=`Spectrum update delayed · ${error.message}`;
    } finally {
      if(generation===state.liveGeneration && state.liveView) spectrumPoll=setTimeout(update,delay);
    }
  };
  update();
}

function beginNewRegion() {
  state.editRegion = null;
  el("save-region").textContent = "Save region label";
  el("label-history").hidden = true;
  el("label-history-list").replaceChildren();
  el("label-state").textContent = "Partial labels are welcome.";
  for(const id of ["family","method","protocol","variant","notes"]) el(id).value = "";
}
async function loadLabelHistory(region,captureId) {
  el("label-history").hidden = false;
  el("label-history-list").textContent = "Loading revisions…";
  try {
    const history = await request(`/api/captures/${captureId}/regions/${region.id}/history`);
    if(state.capture?.id!==captureId || state.editRegion?.id!==region.id) return;
    const list=el("label-history-list");list.replaceChildren();
    for(const revision of history) {
      const item=document.createElement("p");
      item.textContent=`Revision ${revision.revision} · ${revision.source} / ${revision.review_state} · ${[revision.family,revision.method,revision.protocol,revision.variant].filter(Boolean).join(" / ") || "Unknown"}${revision.notes ? ` · ${revision.notes}` : ""}`;
      list.append(item);
    }
  } catch(error) {
    if(state.editRegion?.id===region.id) el("label-history-list").textContent=`History unavailable · ${error.message}`;
  }
}

function updateOutputContext() {
  let text = state.textOutputContext || "Shared Fldigi output · audio route external";
  if (state.outputTab === "images") text = "Received image library · signal association unavailable";
  if (state.outputTab === "raw") {
    const mode = el("raw-tool").value;
    const artifact = state.demodExports[mode];
    text = artifact ? `${mode.toUpperCase()} · capture ${artifact.source_capture.capture_id.slice(0,8)} · ${(artifact.settings.sample_start/artifact.source_capture.sample_rate).toFixed(2)}–${((artifact.settings.sample_start+artifact.settings.sample_count)/artifact.source_capture.sample_rate).toFixed(2)} s · protocol unknown` : "Raw analysis · no result for this tool";
  }
  el("output-context").textContent = text;
}

function renderRegionShortcuts() {
  const list=el("signal-region-shortcuts");list.replaceChildren();
  for(const region of state.regions) {
    const button=document.createElement("button");button.type="button";
    button.className="region-shortcut";
    button.textContent=`${region.variant || region.protocol || region.method || region.family || "Unknown"} · ${(region.sample_start/state.capture.sample_rate).toFixed(2)} s`;
    button.addEventListener("click",()=>selectSavedRegion(region));list.append(button);
  }
  if(!state.regions.length) list.textContent="No saved annotations in this recording.";
}
