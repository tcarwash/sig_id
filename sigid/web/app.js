import { liveTimeLabels } from "/waterfall-window.mjs";
import { decodeSpectrumPacket, LiveWaterfall, liveDisplayBounds } from "/live-waterfall.mjs";
import { createDemodExport, downloadDemodExport } from "/demod-export.mjs";
import { bitSelectionRange, bitstreamMetrics, bytesForBits, framingCandidates, transformBitstream, createBitAnnotation } from "/bit-inspector.mjs";
import { narrowKnownModes } from "/mode-candidates.mjs";
import { layoutSignalLabels, signalCaption, hitSignal } from "/signal-overlays.mjs";

const el = (id) => document.getElementById(id);
const state = { capture: null, waterfall: null, drag: null, selection: null, regions: [], tracks: [], cvDetections: [], cvDismissedCount: 0, cvAvailable: false, cvSimilarAvailable: false, bitAnnotations: [], protocolStencils: [], liveView: false, liveSessionId: null, modelAvailable: false, view: null, demodExports: {}, selectedTrack: null, classifierProposal: null, frozen: false, labelModeId: null, modeSearchResults: [], modeSearchRequest: 0, applyingMode: false, adjustingGeometry: false, geometryDraft: false, splitMode: false, splitChildren: [], splitParentSelection: null, mergeRegionIds: new Set(), modeProposalRegion: null, proposalCaptureRequest: 0, proposalEvidenceCaptures: [] };
state.layaAvailable = false;
state.coverageWindow = null;
state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
state.coverageMaskMode = null;
state.cvCoverageQueue = null;
state.cvReviewWindowSeconds = 10;
state.cvReviewMinSamples = 4096;
state.cvReviewMaxFrequencyHz = 3000;
let lastExpandedOutputHeight = 180;
const baseCanvas = el("waterfall");
const overlayCanvas = el("overlay");
let livePoll = null;
let spectrumPoll = null;
let liveDisplay = null;
let fldigiPoll = null;
let fldigiOffset = null;
let watchedTrialTrack = null;
let isolatedTrialActive = false;
let selectedAudioPlayback = null;

const DEMOD_EXPORT_BUTTONS = {
  "2fsk": "demod-export-2fsk",
  ook: "demod-export-ook",
  mfsk: "demod-export-mfsk",
  bpsk: "demod-export-bpsk",
  qpsk: "demod-export-qpsk",
};

function setOutputDockHeight(height, persist = false) {
  const dock = el("output-dock");
  const resizeHandle = el("output-resize");
  const minimum = 90;
  const maximum = Math.max(minimum, Math.floor(window.innerHeight * 0.72));
  const next = Math.max(minimum, Math.min(maximum, Math.round(height)));
  lastExpandedOutputHeight = next;
  dock.style.height = `${next}px`;
  document.documentElement.style.setProperty("--output-dock-height", `${next}px`);
  resizeHandle.setAttribute("aria-valuemax", String(maximum));
  resizeHandle.setAttribute("aria-valuenow", String(next));
  if (persist) {
    try { localStorage.setItem("sig-id.output-dock-height", String(next)); }
    catch { /* The pane still resizes when storage is unavailable. */ }
  }
  return next;
}

try {
  const savedHeight = Number(localStorage.getItem("sig-id.output-dock-height"));
  if (Number.isFinite(savedHeight) && savedHeight >= 90) setOutputDockHeight(savedHeight);
} catch { /* The default pane height remains usable when storage is unavailable. */ }

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

function renderRawBitInspector(mode, result) {
  const output = el(mode === "2fsk" ? "demod-output" : `${mode}-output`);
  const hypotheses = mode === "bpsk" ? [{ name: "normal", bits: result.bits }, { name: "inverted", bits: result.inverted_bits }]
    : mode === "qpsk" ? (result.symbol_hypotheses || []).map((item, index) => ({ name: `rotation-${index + 1}`, bits: item.bits }))
      : [{ name: "bits", bits: result.bits }];
  const valid = hypotheses.filter((item) => typeof item.bits === "string" && /^[01]+$/.test(item.bits));
  if (!valid.length) return;
  const summary = output.textContent;
  output.replaceChildren();
  const intro = document.createElement("div"); intro.className = "decoded-text"; intro.textContent = summary; output.append(intro);
  const modulation = ({ "2fsk": "2-FSK", ook: "OOK", mfsk: "MFSK", bpsk: "BPSK", qpsk: "QPSK" })[mode];
  const candidates = narrowKnownModes({ modulation });
  const shortlist = document.createElement("div"); shortlist.className = "mode-shortlist";
  const shortlistTitle = document.createElement("strong"); shortlistTitle.textContent = `Possible Fldigi modes · ${modulation} · reviewed subset`; shortlist.append(shortlistTitle);
  if (candidates.length) {
    const list = document.createElement("ul");
    for (const candidate of candidates) { const item = document.createElement("li"); item.textContent = `${candidate.mode} — ${candidate.reasons.join("; ")} · ${candidate.evidence}`; list.append(item); }
    shortlist.append(list);
  } else {
    const empty = document.createElement("span"); empty.textContent = "No loopback-validated mode in this modulation family yet."; shortlist.append(empty);
  }
  const caveat = document.createElement("small"); caveat.textContent = "Shortlist only: other Fldigi modes may also fit, and this does not confirm an ID."; shortlist.append(caveat); output.append(shortlist);
  for (const hypothesis of valid) {
    const section = document.createElement("section"); section.className = "bit-inspector";
    const title = document.createElement("h4"); title.textContent = `${hypothesis.name} · ${hypothesis.bits.length} recovered bits`; section.append(title);
    const controls = document.createElement("div"); controls.className = "bit-interpret-controls";
    const transformSelect = document.createElement("select"); transformSelect.setAttribute("aria-label", "Bitstream interpretation transform");
    const transforms = [
      ["raw", "Raw bits"],
      ["inverted", "Invert polarity"],
      ["differential-xor", "Differential / NRZI · 1 means transition"],
      ["differential-xnor", "Differential / NRZI · 0 means transition"],
    ];
    for (const [value, text] of transforms) { const option = document.createElement("option"); option.value = value; option.textContent = text; transformSelect.append(option); }
    const initialSelect = document.createElement("select"); initialSelect.setAttribute("aria-label", "Differential initial state");
    for (const value of [0, 1]) { const option = document.createElement("option"); option.value = String(value); option.textContent = `Initial state ${value}`; initialSelect.append(option); }
    const alignmentSelect = document.createElement("select"); alignmentSelect.setAttribute("aria-label", "Byte alignment offset");
    for (let value = 0; value < 8; value += 1) { const option = document.createElement("option"); option.value = String(value); option.textContent = `Byte start at bit ${value}`; alignmentSelect.append(option); }
    controls.append(transformSelect, initialSelect, alignmentSelect);
    section.append(controls);
    const viewContent = document.createElement("div"); section.append(viewContent); output.append(section);

    const renderView = () => {
      viewContent.replaceChildren();
      const transform = transformSelect.value;
      const initialState = Number(initialSelect.value);
      const alignment = Number(alignmentSelect.value);
      const bits = transformBitstream(hypothesis.bits, transform, initialState);
      const interpretationName = transform === "raw" ? hypothesis.name
        : transform === "inverted" ? `${hypothesis.name} · inverted`
          : `${hypothesis.name} · ${transform === "differential-xor" ? "NRZI 1=transition" : "NRZI 0=transition"} · initial ${initialState}`;
      const flags = framingCandidates(bits, state.protocolStencils);
      const metrics = bitstreamMetrics(bits, alignment);
      const metricsLine = document.createElement("p"); metricsLine.className = "bit-metrics";
      const showMetric = (value, digits = 3) => value === null ? "n/a" : value.toFixed(digits);
      const leading = Math.min(alignment, bits.length);
      const trailing = metrics.byte_count ? (bits.length - leading) % 8 : Math.max(0, bits.length - leading);
      metricsLine.textContent = `Byte alignment ${alignment} · skip ${leading} leading bit${leading === 1 ? "" : "s"} · ${metrics.byte_count} complete bytes · ${trailing} trailing bit${trailing === 1 ? "" : "s"} · bit entropy ${showMetric(metrics.bit_entropy_bits)} bits/bit · byte entropy ${showMetric(metrics.byte_entropy_bits)} bits/byte · ones ${showMetric(metrics.one_fraction)} · transitions ${showMetric(metrics.transition_fraction)} · longest run ${metrics.longest_run} · printable MSB/LSB ${showMetric(metrics.printable_byte_fraction)}/${showMetric(metrics.lsb_printable_byte_fraction)}`;
      viewContent.append(metricsLine);
      const caution = document.createElement("small"); caution.className = "bit-order-note";
      caution.textContent = "These statistics are descriptive, not confidence. Pattern hits (including repeats) may overlap or occur by chance; they do not validate framing or identify a protocol.";
      viewContent.append(caution);

      if (flags.length) {
        const clues = document.createElement("div"); clues.className = "bit-clues";
        const groupedFlags = new Map();
        for (const flag of flags) {
          if (!groupedFlags.has(flag.stencil_id)) groupedFlags.set(flag.stencil_id, { flag, offsets: [] });
          groupedFlags.get(flag.stencil_id).offsets.push(flag.bit_start);
        }
        for (const { flag, offsets } of groupedFlags.values()) {
          const clue = document.createElement("span");
          clue.textContent = `Pattern clue: ${flag.label} · ${offsets.length} occurrence${offsets.length === 1 ? "" : "s"} at bit${offsets.length === 1 ? "" : "s"} ${offsets.slice(0, 8).join(", ")}${offsets.length > 8 ? ", …" : ""} · ${flag.evidence}`;
          clues.append(clue);
        }
        viewContent.append(clues);
      }

      const table = document.createElement("div"); table.className = "bit-byte-grid";
      table.setAttribute("role", "table");
      table.setAttribute("aria-label", "Recovered bits with MSB-first and LSB-first hexadecimal and ASCII views");
      const header = document.createElement("div"); header.className = "bit-byte bit-byte-header"; header.setAttribute("role", "row");
      for (const label of ["Bit offset", "Bits", "MSB hex", "MSB ASCII", "LSB hex", "LSB ASCII"]) {
        const cell = document.createElement("span"); cell.textContent = label; cell.setAttribute("role", "columnheader"); header.append(cell);
      }
      table.append(header);
      let anchor = null, selected = null;
      const tagButton = document.createElement("button"); tagButton.type = "submit"; tagButton.textContent = "Tag selected bits"; tagButton.disabled = true;
      const stencilButton = document.createElement("button"); stencilButton.type = "submit"; stencilButton.textContent = "Save stencil"; stencilButton.disabled = true;
      const stencilPattern = document.createElement("input"); stencilPattern.placeholder = "Pattern · 0/1 required, ? wildcard"; stencilPattern.maxLength = 512; stencilPattern.pattern = "[01?]{4,512}"; stencilPattern.setAttribute("aria-label", "Stencil bit pattern");
      const paint = () => {
        for (const row of table.querySelectorAll("[data-bit-start]")) {
          const start = Number(row.dataset.bitStart), count = Number(row.dataset.bitCount);
          const inSelection = selected && start < selected.end && start + count > selected.start;
          const isFlag = flags.some((flag) => start < flag.bit_start + flag.bit_count && start + count > flag.bit_start);
          row.classList.toggle("selected", Boolean(inSelection)); row.classList.toggle("candidate", isFlag);
          const tagged = (state.demodExports[mode]?.bit_annotations || []).find((annotation) =>
            annotation.hypothesis === `${interpretationName} · byte start ${alignment}`
            && start < annotation.bit_start + annotation.bit_count && start + count > annotation.bit_start);
          if (tagged) row.style.setProperty("--tag-color", tagged.color || "#f4bf75");
          else row.style.removeProperty("--tag-color");
        }
      };
      for (const cell of bytesForBits(bits, alignment)) {
        const row = document.createElement("div"); row.className = "bit-byte"; row.setAttribute("role", "row");
        row.dataset.bitStart = cell.bit_start; row.dataset.bitCount = cell.bit_count;
        const offset = document.createElement("span"); offset.className = "bit-offset"; offset.textContent = String(cell.bit_start).padStart(6, "0"); offset.setAttribute("role", "rowheader");
        const binary = document.createElement("button"); binary.type = "button"; binary.className = "bit-value"; binary.textContent = cell.binary.padEnd(8, "·"); binary.title = `Bits ${cell.bit_start}–${cell.bit_start + cell.bit_count - 1}; click two rows to select a range`; binary.setAttribute("role", "cell");
        const hex = document.createElement("span"); hex.className = "bit-hex"; hex.textContent = cell.hex; hex.setAttribute("role", "cell");
        const ascii = document.createElement("span"); ascii.className = "bit-ascii"; ascii.textContent = cell.ascii; ascii.setAttribute("role", "cell");
        const lsbHex = document.createElement("span"); lsbHex.className = "bit-hex"; lsbHex.textContent = cell.lsb_hex; lsbHex.setAttribute("role", "cell");
        const lsbAscii = document.createElement("span"); lsbAscii.className = "bit-ascii"; lsbAscii.textContent = cell.lsb_ascii; lsbAscii.setAttribute("role", "cell");
        binary.addEventListener("click", () => {
          if (anchor === null) { anchor = cell; selected = bitSelectionRange(cell, cell); }
          else { selected = bitSelectionRange(anchor, cell); anchor = null; }
          paint(); tagButton.disabled = !selected; stencilButton.disabled = !selected;
          if (selected) stencilPattern.value = bits.slice(selected.start, selected.end);
        });
        row.append(offset, binary, hex, ascii, lsbHex, lsbAscii); table.append(row);
      }
      paint();

      const tagForm = document.createElement("form"); tagForm.className = "bit-tag-form";
      const label = document.createElement("input"); label.placeholder = "Operator interpretation (e.g. likely header)"; label.setAttribute("aria-label", "Operator interpretation");
      const color = document.createElement("input"); color.type = "color"; color.value = "#f4bf75"; color.setAttribute("aria-label", "Tag color");
      tagForm.append(label, color, tagButton);
      const annotationChips = document.createElement("div"); annotationChips.className = "bit-annotations";
      const renderAnnotationChips = () => {
        annotationChips.replaceChildren();
        for (const annotation of state.demodExports[mode]?.bit_annotations || []) {
          if (annotation.hypothesis !== `${interpretationName} · byte start ${alignment}`) continue;
          const chip = document.createElement("span"); chip.className = "bit-annotation"; chip.style.setProperty("--tag-color", annotation.color || "#f4bf75");
          chip.textContent = `${annotation.label} · bits ${annotation.bit_start}–${annotation.bit_start + annotation.bit_count - 1}`; annotationChips.append(chip);
        }
      };
      tagForm.addEventListener("submit", async (event) => {
        event.preventDefault(); if (!selected || !label.value.trim()) return;
        const artifact = state.demodExports[mode];
        const bitStart = selected.start, bitCount = selected.end - selected.start;
        const hypothesisName = `${interpretationName} · byte start ${alignment}`;
        const annotation = createBitAnnotation({ bit_start: bitStart, bit_count: bitCount, label: label.value, color: color.value, hypothesis: hypothesisName });
        const interpretation = { transform, initial_state: initialState, byte_alignment: alignment, byte_order: "both" };
        annotation.interpretation = interpretation;
        annotation.settings = { ...artifact.settings, interpretation };
        tagButton.disabled = true; tagButton.textContent = "Saving…";
        try {
          const source = artifact.source_capture;
          const persisted = await request(`/api/captures/${source.capture_id}/bit-annotations`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              mode, sample_start: artifact.result.sample_start ?? artifact.settings.sample_start,
              sample_count: artifact.result.sample_count ?? artifact.settings.sample_count,
              settings: annotation.settings,
              hypothesis: hypothesisName, bits,
              bit_start: bitStart, bit_count: bitCount, label: annotation.label, color: annotation.color,
            }),
          });
          Object.assign(annotation, persisted); artifact.bit_annotations.push(annotation); renderAnnotationChips(); paint();
          label.value = "";
          await refreshBitAnnotations(source.capture_id);
        } catch (error) {
          const failure = document.createElement("span"); failure.className = "bit-save-error"; failure.textContent = `Tag not saved: ${error.message}`; tagForm.append(failure);
        } finally { tagButton.disabled = !selected; tagButton.textContent = "Tag selected bits"; }
      });
      const stencilForm = document.createElement("form"); stencilForm.className = "bit-stencil-form";
      const stencilHeading = document.createElement("strong"); stencilHeading.textContent = "Create reusable stencil from selected bits";
      const stencilName = document.createElement("input"); stencilName.placeholder = "Stencil name"; stencilName.maxLength = 100; stencilName.setAttribute("aria-label", "Stencil name");
      const stencilField = document.createElement("input"); stencilField.placeholder = "Field, e.g. preamble"; stencilField.maxLength = 100; stencilField.setAttribute("aria-label", "Stencil field");
      const stencilProtocols = document.createElement("input"); stencilProtocols.placeholder = "Possible protocols, comma separated"; stencilProtocols.setAttribute("aria-label", "Possible protocols");
      const stencilStatus = document.createElement("span"); stencilStatus.className = "bit-stencil-status";
      stencilForm.append(stencilHeading, stencilName, stencilField, stencilPattern, stencilProtocols, stencilButton, stencilStatus);
      stencilForm.addEventListener("submit", async (event) => {
        event.preventDefault(); if (!selected || !state.capture) return;
        stencilButton.disabled = true; stencilButton.textContent = "Saving…";
        try {
          const saved = await request("/api/protocol-stencils", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
            name: stencilName.value, field: stencilField.value, pattern: stencilPattern.value,
            protocols: stencilProtocols.value.split(",").map((value) => value.trim()).filter(Boolean),
            source_capture_id: state.capture.id,
          }) });
          state.protocolStencils.push(saved);
          stencilStatus.textContent = "Saved for future bitstream checks; rerun analysis to apply it here.";
        } catch (error) { stencilStatus.textContent = `Stencil not saved: ${error.message}`; }
        finally { stencilButton.disabled = !selected; stencilButton.textContent = "Save stencil"; }
      });
      renderAnnotationChips();
      viewContent.append(table, tagForm, annotationChips, stencilForm);
    };
    const updateInterpretation = () => {
      initialSelect.disabled = !transformSelect.value.startsWith("differential-");
      renderView();
    };
    transformSelect.addEventListener("change", updateInterpretation);
    initialSelect.addEventListener("change", renderView);
    alignmentSelect.addEventListener("change", renderView);
    updateInterpretation();
  }
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
  if (isolatedTrialActive) return;
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
  if (!result.run_id || watchedTrialTrack === result.track_id) return;
  watchedTrialTrack = result.track_id;
  state.decoderTrial = {track_id:result.track_id,status:result.status,mode:result.current_mode,run_id:result.run_id};
  state.textOutputContext = `Isolated mode sweep ${result.run_id.slice(0,8)} · track ${result.track_id} · tentative`;
  updateOutputContext();
  el("fldigi-status").textContent = `Trying isolated tested modes for ${result.track_id} · exact live sample window`;
  el("decoded-text").textContent = `Isolated receive trial: ${result.current_mode}\nTentative only · ${result.note}`;
  isolatedTrialActive = true;
  pollLiveDecodeRun(result.session_id || state.liveSessionId, result.run_id);
}

async function startDecoderTrial(track) {
  if (!track.track_id || !state.liveView) return;
  try {
    const result = await request(`/api/live/sessions/${encodeURIComponent(state.liveSessionId)}/decode-trials`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track_id: track.track_id,
        carrier_hz: track.center_hz,
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

function stopSelectedAudioPlayback(status = "Selected-region playback stopped.") {
  if (!selectedAudioPlayback) return;
  selectedAudioPlayback = null;
  const audio = el("audio-player");
  audio.pause();
  el("play-selection").textContent = "Replay selected region";
  el("play-selection").setAttribute("aria-pressed", "false");
  if (status) el("audio-region-status").textContent = status;
  updateDecoderSelectionControls();
}

el("play-selection").addEventListener("click", async () => {
  if (selectedAudioPlayback) {
    stopSelectedAudioPlayback();
    return;
  }
  const selection = state.selection;
  const capture = state.capture;
  if (!capture || state.liveView || !selection || !state.waterfall) return;
  const sampleRate = capture.sample_rate;
  const startSeconds = selection.sample_start / sampleRate;
  const requestedEndSeconds = (selection.sample_start + selection.sample_count) / sampleRate;
  const playback = { captureId: capture.id, selection, startSeconds, requestedEndSeconds, endSeconds: null, started: false };
  selectedAudioPlayback = playback;
  el("play-selection").textContent = "Stop region playback";
  el("play-selection").setAttribute("aria-pressed", "true");
  updateDecoderSelectionControls();
  const audio = el("audio-player");
  el("audio-region-status").textContent = `Loading selected audio · ${startSeconds.toFixed(2)}–${requestedEndSeconds.toFixed(2)} s`;
  const fail = (message) => {
    if (selectedAudioPlayback === playback) stopSelectedAudioPlayback(message);
  };
  const begin = async () => {
    if (selectedAudioPlayback !== playback || state.capture?.id !== playback.captureId || state.selection !== playback.selection) return;
    const endSeconds = Math.min(playback.requestedEndSeconds, audio.duration);
    if (!Number.isFinite(audio.duration) || endSeconds <= playback.startSeconds) {
      fail("The selected audio range is outside this recording.");
      return;
    }
    playback.endSeconds = endSeconds;
    try {
      await new Promise((resolve, reject) => {
        const target = playback.startSeconds;
        const finish = () => {
          clearTimeout(timeout);
          audio.removeEventListener("seeked", onSeeked);
          if (Math.abs(audio.currentTime - target) <= 0.02) resolve();
          else reject(new Error("Audio seek did not reach the selected sample offset."));
        };
        const onSeeked = () => finish();
        const timeout = setTimeout(() => {
          audio.removeEventListener("seeked", onSeeked);
          reject(new Error("Audio seek timed out."));
        }, 2000);
        audio.addEventListener("seeked", onSeeked, { once: true });
        audio.currentTime = target;
        if (!audio.seeking && Math.abs(audio.currentTime - target) <= 0.02) finish();
      });
      if (selectedAudioPlayback !== playback) return;
      playback.started = true;
      await audio.play();
      if (selectedAudioPlayback === playback) el("audio-region-status").textContent = `Playing selected audio · ${playback.startSeconds.toFixed(2)}–${endSeconds.toFixed(2)} s`;
    } catch (error) {
      fail(`Audio playback could not start · ${error.message}`);
    }
  };
  if (audio.readyState >= 1 && Number.isFinite(audio.duration)) await begin();
  else {
    audio.addEventListener("loadedmetadata", begin, { once: true });
    audio.addEventListener("error", () => fail("The recording audio could not be loaded."), { once: true });
    audio.load();
  }
});

el("audio-player").addEventListener("timeupdate", () => {
  if (selectedAudioPlayback?.started && el("audio-player").currentTime >= selectedAudioPlayback.endSeconds) {
    stopSelectedAudioPlayback("Selected-region playback finished.");
  }
});
el("audio-player").addEventListener("ended", () => {
  if (selectedAudioPlayback) stopSelectedAudioPlayback("Selected-region playback finished.");
});
el("audio-player").addEventListener("pause", () => {
  if (selectedAudioPlayback?.started && el("audio-player").currentTime < selectedAudioPlayback.endSeconds) {
    stopSelectedAudioPlayback();
  }
});

el("decode-selection").addEventListener("click", async () => {
  const selection = state.selection;
  const captureId = state.capture?.id;
  const liveSessionId = state.liveView ? state.liveSessionId : null;
  const mode = el("fldigi-mode").value;
  if (!selection || (!captureId && !liveSessionId) || !mode) return;
  const carrierText = el("fldigi-carrier-hz").value.trim();
  const carrierHz = carrierText ? Number(carrierText) : null;
  if (carrierText && (!Number.isFinite(carrierHz) || carrierHz <= 0 || carrierHz > 20000)) {
    el("replay-status").textContent = "Enter an audio carrier frequency from 1 to 20000 Hz, or clear the field to leave it unset.";
    return;
  }
  const button = el("decode-selection");
  button.disabled = true;
  const carrierStatus = carrierHz === null ? "carrier unset" : `carrier ${carrierHz.toFixed(0)} Hz`;
  const linkedRegion = state.editRegion && state.regions.find((region) =>
    region.id === state.editRegion.id && region.lineage_state === "active"
      && region.sample_start === selection.sample_start && region.sample_count === selection.sample_count,
  );
  el("replay-status").textContent = `Queuing ${mode} decode · ${carrierStatus} · ${(selection.sample_start / state.waterfall.sample_rate).toFixed(2)}–${((selection.sample_start + selection.sample_count) / state.waterfall.sample_rate).toFixed(2)} s…`;
  try {
    const endpoint = liveSessionId
      ? `/api/live/sessions/${encodeURIComponent(liveSessionId)}/decode`
      : `/api/captures/${captureId}/decode`;
    const result = await request(endpoint, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        mode,
        sample_start: selection.sample_start,
        sample_count: selection.sample_count,
        track_id: state.selectedTrack?.track_id || null,
        carrier_hz: carrierHz,
        region_id: liveSessionId ? null : linkedRegion?.id || null,
      }),
    });
    el("replay-status").textContent = `Decode ${result.status} · ${mode} · ${carrierStatus} · isolated ${liveSessionId ? "live track" : "selected region"} input`;
    if (liveSessionId) pollLiveDecodeRun(liveSessionId, result.run_id);
    else pollSavedDecodeRun(captureId, result.run_id);
  } catch (error) {
    el("replay-status").textContent = `Decode could not start · ${error.message}`;
  } finally {
    updateDecoderSelectionControls();
  }
});

async function pollLiveDecodeRun(sessionId, runId) {
  for (let attempt = 0; attempt < 240; attempt++) {
    if (state.liveSessionId !== sessionId) return;
    try {
      const runs = await request(`/api/live/sessions/${encodeURIComponent(sessionId)}/decode-runs`);
      if (state.liveSessionId !== sessionId) return;
      state.decodeRuns = runs;
      renderDecodeRuns();
      const run = runs.find((item) => item.id === runId);
      if (run) {
        state.decoderTrial = {
          track_id: run.track_id, status: run.status,
          mode: run.current_mode, run_id: run.id,
        };
        el("fldigi-status").textContent = `${run.status} · ${run.attempted_modes.join(" → ")} · ${run.note}`;
        if (!state.frozen) renderSignalLabels();
      }
      if (run && !["queued", "running"].includes(run.status)) {
        el("replay-status").textContent = `${run.status.toUpperCase()} · ${run.current_mode} · ${run.note}`;
        if (run.output_text) {
          showOutput("text");
          state.textOutputContext = `Isolated replay ${run.id.slice(0,8)} · exact live track window · tentative text`;
          el("output-context").textContent = state.textOutputContext;
          el("decoded-text").textContent = run.output_text;
        }
        updateDecoderSelectionControls();
        return;
      }
    } catch (error) {
      el("replay-status").textContent = `Decode status unavailable · ${error.message}`;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  el("replay-status").textContent = "Decode is still running. Stop and reopen the capture later to review its saved result.";
}

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
      el("replay-status").textContent = `Decode status unavailable · ${error.message}`;
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  el("replay-status").textContent = "Decode is still running. Reopen this capture later to review its saved result.";
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

async function refreshCvStatus() {
  try {
    const status = await request("/api/cv/status");
    state.cvAvailable = Boolean(status.available);
    state.cvSimilarAvailable = Boolean(status.similarity_search_available);
    state.cvReviewWindowSeconds = status.window_seconds;
    state.cvReviewMinSamples = status.fft_size;
    state.cvReviewMaxFrequencyHz = status.max_frequency_hz;
    const backend = status.compute_backend || "unknown backend";
    const backendDescription = backend === "rocm-unavailable"
      ? "ROCm runtime unavailable; inference will use CPU"
      : `${backend.toUpperCase()} inference`;
    const heldOut = status.held_out_test;
    const groundTruthBoxes = heldOut ? heldOut.true_positive + heldOut.false_negative : 0;
    const evaluationDescription = !heldOut
      ? "held-out metrics unavailable"
      : groundTruthBoxes
        ? `${heldOut.true_positive}/${groundTruthBoxes} held-out boxes matched at IoU 0.50`
        : "held-out report has no signal boxes";
    const trainingKinds = Array.isArray(status.training_source_kinds) && status.training_source_kinds.length
      ? status.training_source_kinds.join(", ")
      : "not recorded";
    const testKinds = Array.isArray(status.held_out_test_source_kinds) && status.held_out_test_source_kinds.length
      ? status.held_out_test_source_kinds.join(", ")
      : "not recorded";
    el("cv-status").textContent = status.available
      ? `YOLO signal detector loaded · ${backendDescription} · ${evaluationDescription} · train sources: ${trainingKinds} · held-out sources: ${testKinds} · ${status.training_scope || "training scope not recorded"} · ${status.window_seconds}s windows · confidence ${(100 * status.confidence_threshold).toFixed(0)}% · class agnostic`
      : `Optional CV detector unavailable${status.error ? ` · ${status.error}` : " · install the cv extra and configure SIG_ID_CV_MODEL_PATH"}`;
    updateCvButton();
    updateCvSimilarButton();
  } catch (error) {
    state.cvAvailable = false;
    state.cvSimilarAvailable = false;
    el("cv-status").textContent = `CV status unavailable · ${error.message}`;
    updateCvButton();
    updateCvSimilarButton();
  }
}
refreshCvStatus();

function updateCvButton() {
  const button = el("cv-scan");
  if (button) button.disabled = !state.cvAvailable || !state.capture || state.liveView;
}

function updateCvSimilarButton() {
  const button = el("cv-similar-search");
  if (!button) return;
  button.hidden = !state.cvSimilarAvailable || !state.selection || state.liveView;
  button.disabled = button.hidden || (state.selection?.sample_count ?? 0) < 4096;
  el("cv-similar-status").textContent = state.cvSimilarAvailable
    ? "Nearest saved signal patches · candidates only"
    : "Build and configure a CV embedding index to retrieve similar saved signals.";
}

el("cv-similar-search").addEventListener("click", async () => {
  const capture = state.capture;
  const selection = state.selection;
  if (!capture || !selection || state.liveView || !state.cvSimilarAvailable) return;
  const button = el("cv-similar-search");
  button.disabled = true;
  el("cv-similar-status").textContent = "Searching reviewed signal examples…";
  try {
    const result = await request(`/api/captures/${encodeURIComponent(capture.id)}/cv-similar-regions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection.sample_start,
        sample_count: selection.sample_count,
        frequency_low_hz: selection.frequency_low_hz,
        frequency_high_hz: selection.frequency_high_hz,
        exclude_region_id: state.editRegion?.id || null,
        limit: 8,
      }),
    });
    if (state.capture !== capture || state.selection !== selection) return;
    renderCvSimilarResults(result.neighbors);
    el("cv-similar-status").textContent = result.count
      ? `Nearest ${result.count} saved patch${result.count === 1 ? "" : "es"} · similarity is not an identification`
      : "No other reviewed regions are in the embedding index.";
  } catch (error) {
    if (state.capture === capture) el("cv-similar-status").textContent = `Similarity search failed · ${error.message}`;
  } finally {
    updateCvSimilarButton();
  }
});

function renderCvSimilarResults(neighbors) {
  const list = el("cv-similar-results");
  list.replaceChildren();
  for (const neighbor of neighbors) {
    const row = document.createElement("div");
    row.className = "track-row";
    const summary = document.createElement("div");
    summary.className = "track-summary";
    summary.textContent = `${neighbor.label} · cosine ${neighbor.similarity.toFixed(3)}`;
    const detail = document.createElement("div");
    detail.className = "region-detail";
    detail.textContent = `${neighbor.frequency_low_hz.toFixed(0)}–${neighbor.frequency_high_hz.toFixed(0)} Hz · ${(neighbor.sample_start / neighbor.sample_rate).toFixed(2)}–${((neighbor.sample_start + neighbor.sample_count) / neighbor.sample_rate).toFixed(2)} s · ${neighbor.review_state} reference`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Open saved example";
    button.addEventListener("click", async () => {
      try {
        const captures = await request("/api/captures");
        const capture = captures.find((item) => item.id === neighbor.capture_id);
        if (!capture) throw new Error("source capture is no longer in the catalog");
        state.capture = capture;
        await loadCapture();
        const region = state.regions.find((item) => item.id === neighbor.region_id);
        if (region) selectSavedRegion(region);
      } catch (error) { showError(`Could not open similar example · ${error.message}`); }
    });
    row.append(summary, detail, button);
    list.append(row);
  }
  if (!neighbors.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = "No similar reviewed examples found.";
    list.append(empty);
  }
}

el("cv-scan").addEventListener("click", async () => {
  if (!state.capture || state.liveView || !state.cvAvailable) return;
  const capture = state.capture;
  const button = el("cv-scan");
  button.disabled = true;
  button.textContent = "Scanning capture…";
  el("cv-status").textContent = "Scanning completed recording in overlapping 10-second spectrogram windows…";
  try {
    const result = await request(`/api/captures/${encodeURIComponent(capture.id)}/cv-detections`, { method: "POST" });
    if (state.capture !== capture || state.liveView) return;
    state.cvDetections = result.detections.map((detection, index) => ({
      ...detection,
      cv_proposal_id: `${result.model_sha256 || "model"}:${index}:${detection.sample_start}`,
    }));
    state.cvDismissedCount = 0;
    renderCvDetections();
    el("cv-status").textContent = `CV proposed ${result.detections.length} signal box${result.detections.length === 1 ? "" : "es"} · overlapping windows deduplicated; provisional, review before labeling or decoding`;
    drawOverlays();
  } catch (error) {
    if (state.capture === capture) el("cv-status").textContent = `CV scan failed · ${error.message}`;
  } finally {
    button.textContent = "Scan capture with CV";
    updateCvButton();
  }
});

function renderCvDetections() {
  const list = el("cv-detection-list");
  if (!list) return;
  list.replaceChildren();
  for (const [index, detection] of state.cvDetections.entries()) {
    const row = document.createElement("div");
    row.className = "track-row";
    const summary = document.createElement("div");
    summary.className = "track-summary";
    const supportingWindows = (detection.supporting_windows || []).length;
    const corroboration = supportingWindows > 1 ? ` · seen in ${supportingWindows} windows` : "";
    summary.textContent = `CV box ${index + 1} · ${(detection.confidence * 100).toFixed(1)}% detector score${corroboration} · identity unknown`;
    const detail = document.createElement("div");
    detail.className = "region-detail";
    detail.textContent = `${detection.frequency_low_hz.toFixed(0)}–${detection.frequency_high_hz.toFixed(0)} Hz · ${(detection.sample_start / detection.sample_rate).toFixed(2)}–${((detection.sample_start + detection.sample_count) / detection.sample_rate).toFixed(2)} s`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Review selection";
    button.addEventListener("click", () => selectCvDetection(detection));
    const reject = document.createElement("button");
    reject.type = "button";
    reject.textContent = "Reject box";
    reject.title = "Remove this proposal from the current scan; this does not create a training label.";
    reject.addEventListener("click", () => {
      state.cvDetections = state.cvDetections.filter((item) => item !== detection);
      state.cvDismissedCount += 1;
      if (state.selectedTrack?.cv_proposal_id === detection.cv_proposal_id) {
        state.selection = null;
        beginNewRegion();
        el("selection-summary").textContent = "No selection";
        el("selection-details").textContent = "The dismissed CV proposal was cleared from the label editor.";
        el("inspector-context").textContent = "Select a signal region to inspect or label it.";
      }
      renderCvDetections();
      drawOverlays();
      el("cv-status").textContent = state.cvDetections.length
        ? `${state.cvDetections.length} CV proposal${state.cvDetections.length === 1 ? "" : "s"} remain · ${state.cvDismissedCount} rejected from this scan`
        : `All CV proposals rejected from this scan · ${state.cvDismissedCount} dismissed; no training label was recorded`;
    });
    row.append(summary, detail, button, reject);
    list.append(row);
  }
  if (!state.cvDetections.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = state.cvDismissedCount
      ? `No proposals remain · ${state.cvDismissedCount} rejected from this scan; dismissals are not training labels.`
      : "No boxes proposed. The detector may miss weak or unfamiliar signals.";
    list.append(empty);
  }
}

function selectCvDetection(detection) {
  const data = state.waterfall;
  if (!data || state.liveView) return;
  const proposal = {
    sample_start: detection.sample_start,
    sample_count: detection.sample_count,
    frequency_low_hz: detection.frequency_low_hz,
    frequency_high_hz: detection.frequency_high_hz,
    cv_proposal_id: detection.cv_proposal_id,
    cv_model_version: detection.model_version,
    track_state: "CV proposal",
    modulation_hypothesis: "unknown signal",
    tone_count_candidate: "unknown",
    time_start_seconds: detection.sample_start / data.sample_rate,
    time_end_seconds: (detection.sample_start + detection.sample_count) / data.sample_rate,
    detector_evidence: {
      detector: "yolo",
      model_sha256: detection.model_sha256,
      model_version: detection.model_version,
      confidence: detection.confidence,
      box_xyxy_normalized: detection.box_xyxy_normalized,
      image_width: detection.image_width,
      image_height: detection.image_height,
      preprocessing_version: detection.preprocessing_version,
      window_sample_start: detection.window_sample_start,
      window_sample_count: detection.window_sample_count,
      sample_rate: detection.sample_rate,
      supporting_windows: detection.supporting_windows || [],
    },
  };
  selectTrack(proposal);
  el("cv-presence-row").hidden = false;
  el("cv-presence-confirmed").checked = false;
  el("inspector-context").textContent = `Provisional CV box · ${(detection.confidence * 100).toFixed(1)}% detector score · signal identity unknown`;
  el("selection-details").textContent += " · inspect bounds, then run the existing classifier or save as unknown";
}

async function refreshLayaStatus() {
  try {
    const provider = await request("/api/laya/status");
    state.layaAvailable = Boolean(provider.available);
    el("laya-status").textContent = state.layaAvailable
      ? `Laya System One ready · ${provider.reference_examples} labeled synthetic train references · sends selected-region features when clicked`
      : "Laya unavailable · configure LAYA_ENDPOINT";
  } catch (error) {
    state.layaAvailable = false;
    el("laya-status").textContent = `Laya status unavailable · ${error.message}`;
  }
  updateLayaButton();
}
refreshLayaStatus();

function updateModelButton() {
  const tooShort = (state.selection?.sample_count ?? 0) < 1024;
  el("model-propose").disabled = !state.modelAvailable || !state.capture || !state.selection || state.liveView || tooShort;
  el("model-propose").title = tooShort ? "The classifier needs at least 1024 audio samples in this selection." : "Suggest a modulation method for this selection.";
  updateCvSimilarButton();
  updateLayaButton();
}

function updateLayaButton() {
  el("laya-propose").disabled = !state.layaAvailable || !state.capture || !state.selection || (state.selection.sample_count ?? 0) < 1024 || state.liveView || state.coverageOnly || state.adjustingGeometry;
}

el("laya-propose").addEventListener("click", async () => {
  if (!state.capture || !state.selection || state.liveView || !state.layaAvailable || state.coverageOnly) return;
  const selection = state.selection;
  const sourceCapture = state.capture;
  const button = el("laya-propose");
  button.disabled = true;
  button.textContent = "Asking Laya…";
  el("label-state").textContent = "Sending selected-region measurements to Laya for a provisional method judgment…";
  try {
    const result = await request(`/api/captures/${sourceCapture.id}/classify/laya`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection.sample_start,
        sample_count: Math.min(1_000_000, selection.sample_count),
        frequency_low_hz: selection.frequency_low_hz ?? null,
        frequency_high_hz: selection.frequency_high_hz ?? null,
      }),
    });
    if (state.capture !== sourceCapture || state.selection !== selection) return;
    if (result.choice === "unknown") {
      el("label-state").textContent = "Laya abstained · insufficient or mixed evidence; no label changed";
    } else {
      el("family").value = result.family || "";
      el("method").value = result.method || "";
      el("protocol").value = "";
      el("variant").value = "";
      const evidence = `Laya System One proposed ${result.method} from selected-region measurements. Protocol and variant remain unknown; review before saving.`;
      el("notes").value = [el("notes").value.trim(), evidence].filter(Boolean).join(" ");
      el("label-state").textContent = `Laya proposed ${result.method} · review then save to accept`;
    }
  } catch (error) {
    el("label-state").textContent = `Laya identification failed · ${error.message}`;
  } finally {
    button.textContent = "Auto ID with Laya";
    updateLayaButton();
  }
});

el("model-propose").addEventListener("click", async () => {
  if (!state.capture || !state.selection || state.liveView || !state.modelAvailable) return;
  const selection = state.selection;
  const sourceCapture = state.capture;
  state.classifierProposal = null;
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
    state.classifierProposal = {
      capture_id: sourceCapture.id,
      selection,
      result: { ...result, selection_sample_count: selection.sample_count },
    };
    if (!result.family && !result.method) {
      el("label-state").textContent = result.novelty_rejected
        ? `Model abstained · outside trained modulation range (distance ${result.novelty_distance.toFixed(2)} > ${result.novelty_threshold.toFixed(2)})`
        : `Model abstained · top guess ${result.top_candidate} at ${(100 * result.confidence).toFixed(1)}%`;
      if (state.cvSimilarAvailable) {
        el("label-state").textContent += " · Find similar signal examples to review nearest known regions, or leave identity unknown.";
      }
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
  stopSelectedAudioPlayback(null);
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
    state.cvDetections = [];
    state.cvDismissedCount = 0;
    renderCvDetections();
    updateCvButton();
    state.liveView = true;
    state.liveSessionId = started.session_id;
    resetOutputSession();
    state.waterfall = null;
    state.view = null;
    state.selection = null;
    state.regions = [];
    state.tracks = [];
    state.decodeRuns = [];
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
    renderDecodeRuns();
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
        if (analysis.automatic_trial?.run_id) watchDecoderTrial(analysis.automatic_trial);
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
    state.liveSessionId = null;
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

function clearModeSelection(clearSearch = true) {
  state.labelModeId = null;
  if (clearSearch && !state.applyingMode) el("mode-search").value = "";
  el("mode-resolution").textContent = "Partial labels are welcome; unknown attributes stay blank.";
}

function applyModeSelection(mode) {
  state.applyingMode = true;
  state.labelModeId = mode.id;
  el("mode-search").value = mode.display_name;
  for (const field of ["family", "method", "protocol", "variant"]) el(field).value = mode[field] || "";
  el("mode-resolution").textContent = `${mode.display_name} → ${mode.family} / ${mode.method} / ${mode.protocol} / ${mode.variant} · ${mode.status || "curated"} · registry ${mode.registry_version}`;
  state.applyingMode = false;
}

el("mode-search").addEventListener("input", async () => {
  const query = el("mode-search").value.trim();
  clearModeSelection(false);
  const requestId = ++state.modeSearchRequest;
  const options = el("mode-results");
  options.replaceChildren();
  if (query.length < 2) return;
  try {
    const result = await request(`/api/modes?q=${encodeURIComponent(query)}`);
    if (requestId !== state.modeSearchRequest) return;
    state.modeSearchResults = result.modes;
    for (const mode of result.modes) {
      const option = document.createElement("option");
      option.value = mode.display_name;
      option.label = `${mode.family} · ${mode.method} · ${mode.protocol}${mode.status ? ` · ${mode.status}` : ""}`;
      options.append(option);
    }
    el("mode-resolution").textContent = result.modes.length
      ? `${result.modes.length} matching canonical modes · choose a specific result to fill the taxonomy.`
      : "No canonical match · keep a partial or unknown label if appropriate.";
  } catch (error) {
    if (requestId === state.modeSearchRequest) el("mode-resolution").textContent = `Mode lookup unavailable: ${error.message}`;
  }
});

el("mode-search").addEventListener("change", () => {
  const query = el("mode-search").value.trim().toLocaleLowerCase();
  const exact = state.modeSearchResults.find((mode) => mode.display_name.toLocaleLowerCase() === query || mode.id.toLocaleLowerCase() === query);
  if (exact) applyModeSelection(exact);
});

for (const field of ["family", "method", "protocol", "variant"]) {
  el(field).addEventListener("input", () => {
    if (state.applyingMode) return;
    if (state.labelModeId) clearModeSelection();
  });
}

function setModeProposalSource(region = null) {
  const eligible = Boolean(region && state.capture && region.lineage_state !== "superseded");
  state.modeProposalRegion = eligible ? { capture_id: state.capture.id, region_id: region.id } : null;
  el("mode-proposal-submit").disabled = !eligible;
  el("mode-proposal-source-one").textContent = eligible
    ? `Evidence source 1 · ${state.capture.source_metadata.uploaded_filename || state.capture.id} · samples ${region.sample_start}–${region.sample_start + region.sample_count} · SHA-256 ${state.capture.sha256}`
    : "Select an active saved region first to attach it as the first evidence source.";
}

async function refreshModeProposalList() {
  try {
    const proposals = await request("/api/mode-proposals");
    const list = el("mode-proposal-list");
    list.replaceChildren();
    if (!proposals.length) { list.textContent = "No user-proposed modes yet."; return; }
    for (const proposal of proposals) {
      const card = document.createElement("article");
      const title = document.createElement("strong");
      title.textContent = `${proposal.display_name} · ${proposal.status.replaceAll("_", " ")}`;
      const summary = document.createElement("p");
      summary.textContent = `${proposal.id} · ${proposal.family}/${proposal.method}/${proposal.protocol}/${proposal.variant} · ${proposal.evidence_regions.length} evidence regions`;
      const rationale = document.createElement("p");
      rationale.textContent = proposal.distinguishing_evidence;
      card.append(title, summary, rationale);
      if (proposal.status === "pending") {
        for (const [decision, caption] of [["accepted", "Accept locally"], ["rejected", "Reject"]]) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = caption;
          button.addEventListener("click", async () => {
            const note = el("mode-proposal-review-note").value.trim();
            if (note.length < 40) { el("mode-proposal-status").textContent = "Add at least 40 characters explaining the review decision."; return; }
            button.disabled = true;
            try {
              await request(`/api/mode-proposals/${encodeURIComponent(proposal.id)}/decision`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ decision, note }),
              });
              el("mode-proposal-status").textContent = decision === "accepted"
                ? "Mode accepted into this local registry; this is not peer verification."
                : "Mode proposal rejected with review rationale saved.";
              await refreshModeProposalList();
            } catch (error) {
              el("mode-proposal-status").textContent = `Review failed: ${error.message}`;
              button.disabled = false;
            }
          });
          card.append(button);
        }
      }
      list.append(card);
    }
  } catch (error) {
    el("mode-proposal-list").textContent = `Could not load mode proposals: ${error.message}`;
  }
}

async function refreshModeEvidenceCaptures() {
  const select = el("mode-proposal-capture-2");
  try {
    const captures = await request("/api/captures");
    state.proposalEvidenceCaptures = captures;
    select.replaceChildren(new Option("Choose a recording", ""));
    for (const capture of captures) {
      const label = capture.source_metadata.uploaded_filename || capture.source_metadata.generator || capture.id;
      select.add(new Option(`${label} · ${capture.sha256.slice(0, 10)}`, capture.id));
    }
  } catch (error) {
    el("mode-proposal-status").textContent = `Could not load recordings for evidence: ${error.message}`;
  }
}

el("mode-proposal-capture-2").addEventListener("change", async () => {
  const captureId = el("mode-proposal-capture-2").value;
  const regionSelect = el("mode-proposal-region-2");
  regionSelect.replaceChildren(new Option("Choose a region", ""));
  if (!captureId) return;
  const requestId = ++state.proposalCaptureRequest;
  try {
    const regions = await request(`/api/captures/${captureId}/regions`);
    if (requestId !== state.proposalCaptureRequest) return;
    for (const region of regions.filter((item) => item.lineage_state !== "superseded")) {
      const sampleRate = state.proposalEvidenceCaptures.find((capture) => capture.id === captureId)?.sample_rate || 1;
      const time = (region.sample_start / sampleRate).toFixed(2);
      const label = [region.variant, region.protocol, region.method, region.family].find(Boolean) || "Unknown signal";
      regionSelect.add(new Option(`${label} · ${time}s · ${region.id.slice(0, 8)}`, region.id));
    }
  } catch (error) {
    el("mode-proposal-status").textContent = `Could not load evidence regions: ${error.message}`;
  }
});

el("mode-proposal-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.modeProposalRegion) return;
  const captureId = el("mode-proposal-capture-2").value;
  const regionId = el("mode-proposal-region-2").value;
  const button = el("mode-proposal-submit");
  button.disabled = true;
  el("mode-proposal-status").textContent = "Validating source references and submitting proposal…";
  try {
    const result = await request("/api/mode-proposals", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: el("mode-proposal-id").value.trim(),
        display_name: el("mode-proposal-name").value.trim(),
        family: el("mode-proposal-family").value.trim(),
        method: el("mode-proposal-method").value.trim(),
        protocol: el("mode-proposal-protocol").value.trim(),
        variant: el("mode-proposal-variant").value.trim(),
        aliases: el("mode-proposal-aliases").value.split(",").map((value) => value.trim()).filter(Boolean),
        rationale: el("mode-proposal-rationale").value.trim(),
        distinguishing_evidence: el("mode-proposal-distinguishing").value.trim(),
        evidence_regions: [
          { ...state.modeProposalRegion, evidence_type: el("mode-proposal-type-1").value, observations: el("mode-proposal-observations-1").value.trim() },
          { capture_id: captureId, region_id: regionId, evidence_type: el("mode-proposal-type-2").value, observations: el("mode-proposal-observations-2").value.trim() },
        ],
      }),
    });
    el("mode-proposal-status").textContent = `${result.display_name} saved as a pending proposal. It will not appear in mode search until locally reviewed.`;
    await refreshModeProposalList();
  } catch (error) {
    el("mode-proposal-status").textContent = `Proposal not saved: ${error.message}`;
  } finally {
    button.disabled = !state.modeProposalRegion;
  }
});

refreshModeEvidenceCaptures();
refreshModeProposalList();
el("mode-proposal-details").addEventListener("toggle", () => {
  if (el("mode-proposal-details").open) {
    refreshModeEvidenceCaptures();
    refreshModeProposalList();
  }
});

async function refreshBitAnnotations(captureId = state.capture?.id) {
  const container = el("saved-bit-annotations");
  if (!container || !captureId) return;
  const annotations = await request(`/api/captures/${captureId}/bit-annotations`);
  if (state.capture?.id !== captureId) return;
  state.bitAnnotations = annotations;
  container.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = `Saved operator bit tags · ${annotations.length}`;
  container.append(heading);
  if (!annotations.length) {
    const empty = document.createElement("span"); empty.textContent = "No saved bit tags for this capture."; container.append(empty); return;
  }
  for (const annotation of annotations) {
    const chip = document.createElement("span"); chip.className = "bit-annotation";
    chip.style.setProperty("--tag-color", annotation.color);
    chip.textContent = `${annotation.label} · ${annotation.mode}/${annotation.hypothesis} · bits ${annotation.bit_start}–${annotation.bit_start + annotation.bit_count - 1}`;
    chip.title = `Capture samples ${annotation.sample_start}–${annotation.sample_start + annotation.sample_count - 1} · ${annotation.stream_sha256}`;
    container.append(chip);
  }
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
  stopSelectedAudioPlayback();
  const generation = state.captureGeneration = (state.captureGeneration || 0) + 1;
  clearDemodExports();
  state.liveView = false;
  state.liveSessionId = null;
  resetOutputSession();
  state.coverageRecords = [];
  state.cvCoverageQueue = null;
  el("cv-coverage-queue-status").textContent = "Save signal boxes first. A complete window without boxes becomes a negative training example.";
  state.coverageEdit = null;
  state.coverageOnly = false;
  state.coverageWindow = null;
  state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
  state.coverageMaskMode = null;
  el("coverage-panel").hidden = true;
  state.editRegion = null;
  state.classifierProposal = null;
  el("start-scene-review").disabled = true;
  updateCoverageMaskControls();
  state.waterfall = null;
  updateSceneReviewAvailability();
  baseCanvas.getContext("2d").clearRect(0, 0, baseCanvas.width, baseCanvas.height);
  overlayCanvas.getContext("2d").clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  el("signal-labels")?.replaceChildren();
  state.view = null;
  state.selection = null;
  state.cvDetections = [];
  state.cvDismissedCount = 0;
  el("cv-similar-results")?.replaceChildren();
  el("cv-similar-status").textContent = "";
  renderCvDetections();
  updateDecoderSelectionControls();
  updateModelButton();
  updateCvButton();
  updateCvSimilarButton();
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
  updateCvButton();
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
  updateSceneReviewAvailability();
  const regions = await request(`/api/captures/${c.id}/regions`);
  if (generation !== state.captureGeneration || state.liveView) return;
  state.regions = regions;
  renderRegions();
  await refreshSceneCoverage(c.id);
  try { await refreshBitAnnotations(c.id); }
  catch (error) { el("saved-bit-annotations").textContent = `Saved bit tags unavailable · ${error.message}`; }
  try { state.protocolStencils = await request("/api/protocol-stencils"); }
  catch (error) { state.protocolStencils = []; showError(`Saved protocol stencils unavailable · ${error.message}`); }
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

async function refreshSceneCoverage(captureId = state.capture?.id) {
  if (!captureId) return null;
  try {
    const records = await request(`/api/captures/${captureId}/coverage`);
    if (state.capture?.id !== captureId) return null;
    state.coverageRecords = records;
    renderSceneCoverage();
    return records;
  } catch (error) {
    if (state.capture?.id === captureId) {
      el("coverage-list").textContent = `Scene reviews unavailable · ${error.message}`;
    }
    return null;
  }
}

function updateSceneReviewAvailability() {
  const selection = state.selection;
  const valid = Boolean(state.capture && !state.liveView && !state.coverageOnly && selection && Number.isInteger(selection.sample_start) && Number.isInteger(selection.sample_count) && selection.sample_count > 0 && !state.adjustingGeometry && !state.splitMode);
  el("start-scene-review").disabled = !valid;
  el("start-cv-coverage-queue").disabled = !state.capture || !state.waterfall || state.liveView || state.capture.sample_rate / 2 < state.cvReviewMaxFrequencyHz;
}

function cvReviewWindows(capture) {
  const windowSamples = capture.sample_rate * state.cvReviewWindowSeconds;
  const windows = [];
  for (let start = 0; start < capture.sample_count; start += windowSamples) {
    const count = Math.min(windowSamples, capture.sample_count - start);
    if (count >= state.cvReviewMinSamples) windows.push({ sample_start: start, sample_count: count });
  }
  return windows;
}

function hasCompleteCvCoverage(window) {
  const end = window.sample_start + window.sample_count;
  return state.coverageRecords.some((record) => {
    if (record.task !== "detection" || record.state !== "complete" || record.is_stale) return false;
    if (record.sample_start > window.sample_start || record.sample_start + record.sample_count < end) return false;
    if (record.frequency_low_hz !== null && (record.frequency_low_hz > 0 || record.frequency_high_hz < state.cvReviewMaxFrequencyHz)) return false;
    return !(record.ignore_regions?.length || record.uncertain_regions?.length);
  });
}

function showCvReviewWindow(window, progress) {
  if (!state.capture || state.liveView) return;
  state.coverageEdit = null;
  state.coverageOnly = false;
  state.selection = {
    ...window,
    frequency_low_hz: 0,
    frequency_high_hz: state.cvReviewMaxFrequencyHz,
  };
  beginNewRegion();
  updateSceneReviewAvailability();
  updateCarrierFromSelection(state.selection);
  reprojectSelection();
  drawOverlays();
  startSceneCoverageReview();
  el("coverage-task").value = "detection";
  el("coverage-scope").value = "";
  el("coverage-notes").value = "";
  el("coverage-status").textContent = `${progress} · check every signal has an accepted box before marking Complete. Complete empty windows train as negatives. Mark Partial if unsure; add/correct boxes and restart the queue.`;
}

function nextCvReviewWindow() {
  const queue = state.cvCoverageQueue;
  if (!queue || state.capture?.id !== queue.captureId) return;
  const next = queue.windows.find((window) => !queue.visited.has(window.sample_start) && !hasCompleteCvCoverage(window));
  if (!next) {
    const pendingPartial = queue.windows.filter((window) => !hasCompleteCvCoverage(window)).length;
    const message = pendingPartial
      ? `Review queue finished · ${queue.alreadyComplete + queue.completeCount}/${queue.totalWindows} complete, ${queue.partialCount} partial/uncertain; ${pendingPartial} windows still need complete review.`
      : `Review queue finished · all ${queue.totalWindows} CV windows have fresh complete detection coverage.`;
    state.cvCoverageQueue = null;
    exitSceneCoverageReview();
    el("cv-coverage-queue-status").textContent = message;
    return;
  }
  const progress = `${queue.alreadyComplete + queue.visited.size + 1}/${queue.totalWindows} accounted for · ${queue.completeCount} complete, ${queue.partialCount} partial/uncertain this pass`;
  el("cv-coverage-queue-status").textContent = progress;
  showCvReviewWindow(next, progress);
}

async function startCvCoverageQueue() {
  const capture = state.capture;
  if (!capture || !state.waterfall || state.liveView || capture.sample_rate / 2 < state.cvReviewMaxFrequencyHz) return;
  const records = await refreshSceneCoverage(capture.id);
  if (state.capture !== capture) return;
  if (!records) {
    el("cv-coverage-queue-status").textContent = "Could not load scene reviews; retry after checking the workbench connection.";
    return;
  }
  state.coverageRecords = records;
  const windows = cvReviewWindows(capture);
  const pending = windows.filter((window) => !hasCompleteCvCoverage(window));
  if (!pending.length) {
    state.cvCoverageQueue = null;
    el("cv-coverage-queue-status").textContent = `All ${windows.length} CV windows have fresh complete detection coverage.`;
    return;
  }
  state.cvCoverageQueue = {
    captureId: capture.id,
    windows: pending,
    totalWindows: windows.length,
    alreadyComplete: windows.length - pending.length,
    visited: new Set(),
    completeCount: 0,
    partialCount: 0,
  };
  nextCvReviewWindow();
}

function updateCoverageMaskControls() {
  const canMark = Boolean(state.coverageOnly && state.coverageWindow && !state.liveView);
  const busy = Boolean(state.coverageMaskMode);
  el("mark-coverage-uncertain").disabled = !canMark || busy || state.coverageMasks.uncertain_regions.length >= 64;
  el("mark-coverage-ignore").disabled = !canMark || busy || state.coverageMasks.ignore_regions.length >= 64;
  el("cancel-coverage-mask").hidden = !state.coverageMaskMode;
}

function renderSceneCoverageMasks() {
  const list = el("coverage-mask-list");
  list.replaceChildren();
  for (const [field, label] of [["uncertain_regions", "Uncertain"], ["ignore_regions", "Ignored"]]) {
    for (const [index, mask] of state.coverageMasks[field].entries()) {
      const item = document.createElement("div");
      item.className = "coverage-row";
      const start = (mask.sample_start / state.capture.sample_rate).toFixed(2);
      const end = ((mask.sample_start + mask.sample_count) / state.capture.sample_rate).toFixed(2);
      const frequency = mask.frequency_low_hz === null
        ? "frequency follows scene scope"
        : `${mask.frequency_low_hz.toFixed(0)}–${mask.frequency_high_hz.toFixed(0)} Hz`;
      const detail = document.createElement("span");
      detail.textContent = `${label} · ${start}–${end} s · ${frequency}${mask.reason ? ` · ${mask.reason}` : ""}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Remove";
      remove.addEventListener("click", () => {
        state.coverageMasks[field].splice(index, 1);
        renderSceneCoverageMasks();
        drawOverlays();
      });
      item.append(detail, remove);
      list.append(item);
    }
  }
  if (!state.coverageMasks.ignore_regions.length && !state.coverageMasks.uncertain_regions.length) {
    list.textContent = "No subregions marked.";
  }
}

function beginCoverageMask(mode) {
  if (!state.coverageOnly || !state.coverageWindow) return;
  if (!el("coverage-mask-reason").value.trim()) {
    el("coverage-mask-status").textContent = "Enter a short reason before drawing this subregion.";
    return;
  }
  if (state.coverageMasks[mode].length >= 64) {
    el("coverage-mask-status").textContent = "This review already has 64 areas of this type.";
    return;
  }
  state.coverageMaskMode = mode;
  el("coverage-mask-status").textContent = `Draw a ${mode === "uncertain_regions" ? "uncertain" : "ignored"} rectangle within the reviewed window.`;
  updateCoverageMaskControls();
}

function cancelCoverageMask() {
  state.coverageMaskMode = null;
  el("coverage-mask-status").textContent = "Area drawing cancelled.";
  updateCoverageMaskControls();
}

function startSceneCoverageReview() {
  if (el("start-scene-review").disabled) return;
  state.coverageEdit = null;
  state.coverageOnly = true;
  updateSceneReviewAvailability();
  state.coverageWindow = { ...state.selection };
  state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
  state.coverageMaskMode = null;
  renderSceneCoverageMasks();
  updateCoverageMaskControls();
  el("coverage-panel").hidden = false;
  el("coverage-notes").value = "";
  el("save-coverage").textContent = "Record reviewed window";
  el("save-region").disabled = true;
  const selection = state.selection;
  const start = selection.sample_start / state.capture.sample_rate;
  const end = (selection.sample_start + selection.sample_count) / state.capture.sample_rate;
  const low = selection.frequency_low_hz;
  const high = selection.frequency_high_hz;
  el("selection-summary").textContent = "Scene coverage review";
  el("selection-details").textContent = `Review this exact window · ${start.toFixed(2)}–${end.toFixed(2)} s · ${Number.isFinite(low) && Number.isFinite(high) ? `${low.toFixed(0)}–${high.toFixed(0)} Hz` : "frequency scope unknown"}`;
  el("coverage-status").textContent = "Check for all in-scope signals; choose Partial / uncertain when the window does not support a complete claim.";
  openInspector("label");
}

function exitSceneCoverageReview() {
  const wasEditing = Boolean(state.coverageEdit);
  state.coverageEdit = null;
  state.coverageOnly = false;
  state.coverageWindow = null;
  state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
  state.coverageMaskMode = null;
  el("coverage-panel").hidden = true;
  el("save-coverage").textContent = "Record reviewed window";
  if (wasEditing) {
    state.selection = null;
    beginNewRegion();
    el("selection-summary").textContent = "No region selected";
    el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
    drawOverlays();
  }
  el("save-region").disabled = !state.selection || !state.capture || state.liveView || Boolean(state.editRegion?.lineage_state === "superseded");
  updateSceneReviewAvailability();
  renderSceneCoverageMasks();
  updateCoverageMaskControls();
}

function renderSceneCoverage() {
  const list = el("coverage-list");
  list.replaceChildren();
  for (const record of state.coverageRecords) {
    const item = document.createElement("div");
    item.className = "coverage-row";
    const start = (record.sample_start / state.capture.sample_rate).toFixed(2);
    const end = ((record.sample_start + record.sample_count) / state.capture.sample_rate).toFixed(2);
    const frequency = record.frequency_low_hz === null
      ? "frequency unknown"
      : `${record.frequency_low_hz.toFixed(0)}–${record.frequency_high_hz.toFixed(0)} Hz`;
    const label = document.createElement("span");
    label.textContent = `${record.task} · ${record.state} · ${start}–${end} s · ${frequency} · r${record.revision}${record.is_stale ? " · STALE" : ""}`;
    const edit = document.createElement("button");
    edit.type = "button";
    edit.textContent = record.is_stale ? "Review again" : "Open review";
    edit.addEventListener("click", () => selectSceneCoverage(record));
    item.append(label, edit);
    list.append(item);
  }
  if (!state.coverageRecords.length) list.textContent = "No scene reviews for this capture yet.";
}

function selectSceneCoverage(record) {
  if (!state.waterfall || !state.capture || state.liveView) return;
  state.selection = {
    sample_start: record.sample_start,
    sample_count: record.sample_count,
    frequency_low_hz: record.frequency_low_hz ?? 0,
    frequency_high_hz: record.frequency_high_hz ?? Math.min(3000, state.waterfall.sample_rate / 2),
  };
  beginNewRegion();
  state.coverageEdit = {
    id: record.id, revision: record.revision,
    frequencyUnknown: record.frequency_low_hz === null,
  };
  state.coverageOnly = true;
  updateSceneReviewAvailability();
  el("coverage-panel").hidden = false;
  state.coverageWindow = { ...state.selection };
  state.coverageMasks = {
    ignore_regions: (record.ignore_regions || []).map((item) => ({ ...item })),
    uncertain_regions: (record.uncertain_regions || []).map((item) => ({ ...item })),
  };
  state.coverageMaskMode = null;
  updateCoverageMaskControls();
  renderSceneCoverageMasks();
  el("coverage-task").value = record.task;
  el("coverage-state").value = record.state;
  el("coverage-scope").value = record.taxonomy_scope.join(", ");
  el("coverage-notes").value = record.notes || "";
  el("save-coverage").textContent = `Save review revision ${record.revision + 1}`;
  el("coverage-status").textContent = record.is_stale
    ? "This scene review is stale because overlapping annotations changed. Review this window again before saving."
    : `Reviewing ${record.state} scene window · annotation digest ${record.annotation_set_digest.slice(0, 12)}…`;
  el("save-region").disabled = true;
  el("save-region").textContent = "Review window selected";
  updateCarrierFromSelection(state.selection);
  reprojectSelection();
  state.coverageWindow = { ...state.selection };
  el("selection-summary").textContent = "Scene review window";
  el("selection-details").textContent = `Samples ${record.sample_start.toLocaleString()}–${(record.sample_start + record.sample_count).toLocaleString()} · ${record.task} review`;
  openInspector("label");
  drawOverlays();
}

function renderDecodeRuns() {
  const list = el("decode-runs");
  list.replaceChildren();
  const runs = state.decodeRuns || [];
  if (!runs.length) {
    list.textContent = state.liveView
      ? "No Fldigi receive runs for this live session yet."
      : "No saved Fldigi receive runs for this capture.";
    return;
  }
  for (const run of runs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "decode-run";
    const interval = Number.isInteger(run.sample_start) && Number.isInteger(run.sample_count) && run.sample_rate
      ? ` · ${(run.sample_start / run.sample_rate).toFixed(2)}–${((run.sample_start + run.sample_count) / run.sample_rate).toFixed(2)} s`
      : " · track time unavailable";
    const carrier = Number.isFinite(run.carrier_hz) ? ` · carrier ${run.carrier_hz.toFixed(0)} Hz` : "";
    const regionLink = run.region_id ? " · linked saved region" : "";
    const classifierLink = run.classifier_evidence?.model_sha256 ? " · classifier evidence saved" : "";
    button.textContent = `${run.current_mode || "Fldigi"} · ${run.status}${carrier}${interval}${regionLink}${classifierLink}`;
    button.addEventListener("click", () => {
      showOutput("text");
      state.textOutputContext = `Saved run ${run.id.slice(0,8)}${run.region_id ? ` · region ${run.region_id.slice(0,8)}` : ""}${run.classifier_evidence?.model_sha256 ? ` · classifier ${run.classifier_evidence.model_sha256.slice(0,8)}` : ""} · ${run.output_scope} · ${run.association}`;
      el("output-context").textContent = state.textOutputContext;
      const classifierSummary = classifierEvidenceSummary(run.classifier_evidence);
      el("decoded-text").textContent = `${run.status.toUpperCase()} · ${run.current_mode || "Fldigi"}\n${run.note}${classifierSummary ? `\n\n${classifierSummary}` : ""}\n\n${run.output_text || "No decoded text was recorded."}`;
      el("decoded-text").scrollTop = 0;
    });
    list.append(button);
  }
}

function classifierEvidenceSummary(evidence) {
  if (!evidence?.model_sha256) return null;
  const proposal = evidence.method
    ? `method ${evidence.method} (${(100 * evidence.confidence).toFixed(1)}%)`
    : `method abstained${evidence.top_candidate ? ` · top candidate ${evidence.top_candidate} (${(100 * evidence.confidence).toFixed(1)}%)` : ""}`;
  const family = evidence.family ? ` · family ${evidence.family} (${(100 * evidence.family_confidence).toFixed(1)}%)` : "";
  return `Classifier evidence: ${proposal}${family} · model ${evidence.model_sha256} · samples ${evidence.sample_start}–${evidence.sample_start + evidence.sample_count}`;
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
  const overlayBounds = (sampleStart, sampleCount, frequencyLow, frequencyHigh) => {
    const sampleEnd = sampleStart + sampleCount;
    if (sampleEnd <= state.view.timeStartSample || sampleStart >= state.view.timeEndSample
      || frequencyHigh <= lowHz || frequencyLow >= highHz) return null;
    const x0 = visibleX(frequencyLow);
    const x1 = visibleX(frequencyHigh);
    const y0 = visibleRow(sampleStart);
    const y1 = visibleRow(sampleEnd);
    const left = Math.max(0, Math.min(x0, x1));
    const right = Math.min(bins - 1, Math.max(x0, x1));
    const top = Math.max(0, Math.min(y0, y1));
    const bottom = Math.min(overlayCanvas.height - 1, Math.max(y0, y1));
    if (right < left || bottom < top) return null;
    return {
      left,
      top,
      width: Math.max(2, right - left),
      height: Math.max(2, bottom - top),
    };
  };
  for (const track of state.tracks) {
    const bounds = overlayBounds(track.sample_start, track.sample_count, track.frequency_low_hz, track.frequency_high_hz);
    if (!bounds) continue;
    ctx.strokeStyle = track.track_state === "persistent" ? "#52d6d0" : "#f1be69";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
  }
  for (const detection of state.cvDetections) {
    const bounds = overlayBounds(detection.sample_start, detection.sample_count,
      detection.frequency_low_hz, detection.frequency_high_hz);
    if (!bounds) continue;
    ctx.strokeStyle = "#bd8cff";
    ctx.lineWidth = 2;
    ctx.setLineDash([2, 3]);
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
  }
  ctx.setLineDash([]);
  for (const region of state.regions) {
    const low = region.frequency_low_hz ?? 0;
    const high = region.frequency_high_hz ?? data.sample_rate / 2;
    const bounds = overlayBounds(region.sample_start, region.sample_count, low, high);
    if (!bounds) continue;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
  }
  if (state.coverageOnly && state.coverageWindow) {
    for (const [field, color] of [["uncertain_regions", "#f1be69"], ["ignore_regions", "#ee796f"]]) {
      for (const mask of state.coverageMasks[field]) {
        const low = mask.frequency_low_hz ?? state.coverageWindow.frequency_low_hz ?? 0;
        const high = mask.frequency_high_hz ?? state.coverageWindow.frequency_high_hz ?? data.sample_rate / 2;
        const bounds = overlayBounds(mask.sample_start, mask.sample_count, low, high);
        if (!bounds) continue;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.2;
        ctx.fillRect(bounds.left, bounds.top, bounds.width, bounds.height);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
      }
    }
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
  stopSelectedAudioPlayback(null);
  const data = state.waterfall;
  if (!data) return;
  if (!state.view) resetWaterfallView(false);
  const view = waterfallViewBounds();
  const lowHz = data.frequency_hz[view.frequencyStart];
  const highHz = data.frequency_hz[view.frequencyEnd - 1];
  const bins = view.frequencyEnd - view.frequencyStart;
  const x1 = Math.max(0, Math.min(bins - 1, Math.round((track.frequency_low_hz - lowHz) / Math.max(1, highHz - lowHz) * (bins - 1))));
  const x2 = Math.max(x1 + 1, Math.min(bins - 1, Math.round((track.frequency_high_hz - lowHz) / Math.max(1, highHz - lowHz) * (bins - 1))));
  const rowA = sampleToRow(track.sample_start);
  const rowB = sampleToRow(track.sample_start + track.sample_count);
  const y1 = Math.max(0, Math.min(rowA, rowB));
  const y2 = Math.min(overlayCanvas.height - 1, Math.max(rowA, rowB));
  const binStep = (highHz - lowHz) / Math.max(1, bins - 1);
  const sampleEnd = Math.min(data.sample_count, track.sample_start + track.sample_count);
  state.selection = { sample_start: track.sample_start, sample_count: Math.max(1, sampleEnd - track.sample_start), frequency_low_hz: track.frequency_low_hz, frequency_high_hz: track.frequency_high_hz, x1, y1, x2, y2 };
  updateCarrierFromSelection(state.selection);
  beginNewRegion();
  state.selectedTrack = track;
  const center = (track.frequency_low_hz + track.frequency_high_hz) / 2;
  const bandwidth = track.frequency_high_hz - track.frequency_low_hz;
  el("inspector-context").textContent = `${track.modulation_hypothesis} candidate · ${center.toFixed(0)} Hz center · ${bandwidth.toFixed(0)} Hz wide · peak SNR estimate ${track.peak_snr_db?.toFixed(1) ?? "unknown"} dB · ${track.tone_count_candidate ?? "unknown"} tone centers${Number.isFinite(track.drift_hz_per_second) ? ` · drift ${track.drift_hz_per_second.toFixed(1)} Hz/s` : ""}`;
  openInspector("label");
  el("selection-summary").textContent = `${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz · ${track.modulation_hypothesis}`;
  el("selection-details").textContent = `Provisional candidate · ${(track.time_start_seconds).toFixed(2)}–${track.time_end_seconds.toFixed(2)} s · ${track.tone_count_candidate} tone centers · protocol unknown`;
  updateSceneReviewAvailability();
  el("family").value = "";
  el("method").value = "";
  el("protocol").value = "";
  el("variant").value = "";
  const trackEvidence = track.track_id ? ` Track ${track.track_id} observed ${track.track_observations} times (${track.track_state}).` : "";
  const snrEvidence = Number.isFinite(track.peak_snr_db) ? ` SNR ${track.peak_snr_db.toFixed(1)} dB.` : "";
  el("notes").value = `Auto-detected candidate (${track.modulation_hypothesis}); review before labeling.${trackEvidence}${snrEvidence}`;
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
  if (!state.waterfall || (state.coverageOnly && !state.coverageMaskMode)) return;
  stopSelectedAudioPlayback(null);
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
  if (moved < 5 && !state.splitMode && !state.coverageMaskMode) {
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
  if (state.coverageMaskMode && state.coverageWindow) {
    const mask = {
      sample_start: state.selection.sample_start,
      sample_count: state.selection.sample_count,
      frequency_low_hz: state.selection.frequency_low_hz,
      frequency_high_hz: state.selection.frequency_high_hz,
      reason: el("coverage-mask-reason").value.trim(),
    };
    const window = state.coverageWindow;
    const withinWindow = mask.sample_start >= window.sample_start
      && mask.sample_start + mask.sample_count <= window.sample_start + window.sample_count
      && mask.frequency_low_hz >= window.frequency_low_hz
      && mask.frequency_high_hz <= window.frequency_high_hz;
    const target = state.coverageMasks[state.coverageMaskMode];
    if (!mask.reason) {
      el("coverage-mask-status").textContent = "Add a short reason before drawing an ignore or uncertain subregion.";
    } else if (!withinWindow) {
      el("coverage-mask-status").textContent = "Mask bounds must stay inside the reviewed scene window.";
    } else {
      target.push(mask);
      el("coverage-mask-status").textContent = `${state.coverageMaskMode === "uncertain_regions" ? "Uncertain" : "Ignored"} subregion added.`;
      renderSceneCoverageMasks();
    }
    state.coverageMaskMode = null;
    state.selection = { ...window };
    state.drag = null;
    updateCoverageMaskControls();
    updateSceneReviewAvailability();
    reprojectSelection();
    drawOverlays();
    return;
  }
  updateCarrierFromSelection(state.selection);
  state.drag = null;

  if (state.splitMode && state.editRegion) {
    const { x1, y1, x2, y2, ...bounds } = state.selection;
    state.splitChildren.push(bounds);
    state.selection = { ...state.splitParentSelection };
    el("save-split").disabled = state.splitChildren.length < 2;
    el("split-status").textContent = `${state.splitChildren.length} child bound${state.splitChildren.length === 1 ? "" : "s"} drawn · draw at least two distinct-signal regions, then save.`;
    drawOverlays();
    return;
  }
  if (state.adjustingGeometry && state.editRegion) {
    state.adjustingGeometry = false;
    state.geometryDraft = true;
    el("save-region-bounds").hidden = false;
    el("cancel-region-bounds").hidden = false;
    el("geometry-status").textContent = `Proposed bounds · samples ${sampleStart.toLocaleString()}–${sampleEnd.toLocaleString()} · ${frequencyLow.toFixed(0)}–${frequencyHigh.toFixed(0)} Hz. Save to append a geometry revision.`;
    el("selection-summary").textContent = "Proposed corrected bounds";
    el("selection-details").textContent = "Original geometry remains in history until these bounds are saved.";
    drawOverlays();
    return;
  }
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
    const classifierProposal = state.classifierProposal?.capture_id === captureId
      && state.classifierProposal.selection === selection
      ? state.classifierProposal.result : null;
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
        mode_id: state.labelModeId,
        origin: !edit && state.selectedTrack ? "detector_proposal" : el("missed-signal").checked && !edit ? "user_added_missed_detection" : "manual",
        detector_evidence: !edit ? state.selectedTrack?.detector_evidence || null : null,
        signal_presence_confirmed: !edit && Boolean(state.selectedTrack?.detector_evidence) && el("cv-presence-confirmed").checked,
        classifier_proposal: classifierProposal ? {
          sample_start: classifierProposal.sample_start,
          sample_count: classifierProposal.sample_count,
          selection_sample_count: classifierProposal.selection_sample_count,
        } : null,
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
    state.classifierProposal = null;
    beginNewRegion();
    renderRegions();
    await refreshSceneCoverage(captureId);
    state.selection = null;
    el("coverage-panel").hidden = true;
    updateModelButton();
    el("selection-summary").textContent = "No region selected";
    el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
    for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = "";
    el("missed-signal").checked = false;
    drawOverlays();
  } catch (error) {
    if (state.capture?.id !== captureId) return;
    showError(error.message);
    if (error.status === 409) {
      state.regions = await request(`/api/captures/${captureId}/regions`);
      renderRegions();
      await refreshSceneCoverage(captureId);
      el("label-state").textContent = "A newer label exists. Reopen its marker to review it; your draft is retained here.";
    }
    button.disabled = false;
  }
});

async function submitAnnotationFeedback(action) {
  const edit = state.editRegion;
  const captureId = state.capture?.id;
  if (!edit || !captureId) return;
  if (action === "incorrect_identity" && !state.labelModeId) {
    el("feedback-status").textContent = "Choose the corrected canonical mode above first.";
    return;
  }
  const buttons = ["feedback-correct", "feedback-wrong", "feedback-not-signal"].map(el);
  buttons.forEach((button) => { button.disabled = true; });
  el("feedback-status").textContent = "Saving feedback…";
  try {
    const result = await request(`/api/captures/${captureId}/regions/${edit.id}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: edit.revision,
        expected_geometry_revision: edit.geometry_revision,
        action,
        ...(action === "incorrect_identity" ? { corrected_mode_id: state.labelModeId } : {}),
      }),
    });
    if (state.capture?.id !== captureId) return;
    state.regions = await request(`/api/captures/${captureId}/regions`);
    renderRegions();
    await refreshSceneCoverage(captureId);
    const updated = state.regions.find((region) => region.id === edit.id);
    const message = `${action.replaceAll("_", " ")} recorded · label revision ${result.resulting_revision}`;
    if (updated) selectSavedRegion(updated);
    el("feedback-status").textContent = message;
  } catch (error) {
    if (state.capture?.id !== captureId) return;
    el("feedback-status").textContent = error.status === 409
      ? "A newer label exists. Reopen the region before submitting feedback."
      : `Feedback failed: ${error.message}`;
  } finally {
    buttons.forEach((button) => { button.disabled = false; });
  }
}

el("feedback-correct").addEventListener("click", () => submitAnnotationFeedback("correct"));
el("feedback-wrong").addEventListener("click", () => submitAnnotationFeedback("incorrect_identity"));
el("feedback-not-signal").addEventListener("click", () => submitAnnotationFeedback("not_signal"));

el("edit-region-bounds").addEventListener("click", () => {
  if (!state.editRegion || !state.selection) return;
  state.adjustingGeometry = true;
  state.geometryDraft = false;
  el("save-region-bounds").hidden = true;
  el("cancel-region-bounds").hidden = false;
  el("geometry-status").textContent = "Drag across the waterfall to propose corrected time/frequency bounds. The saved geometry remains unchanged until you save.";
});

el("save-region-bounds").addEventListener("click", async () => {
  const edit = state.editRegion;
  const captureId = state.capture?.id;
  const selection = state.selection;
  if (!edit || !captureId || !selection || !state.geometryDraft) return;
  const button = el("save-region-bounds");
  button.disabled = true;
  try {
    const result = await request(`/api/captures/${captureId}/regions/${edit.id}/geometry`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        expected_revision: edit.geometry_revision,
        sample_start: selection.sample_start,
        sample_count: selection.sample_count,
        frequency_low_hz: selection.frequency_low_hz,
        frequency_high_hz: selection.frequency_high_hz,
      }),
    });
    if (state.capture?.id !== captureId) return;
    state.regions = await request(`/api/captures/${captureId}/regions`);
    renderRegions();
    await refreshSceneCoverage(captureId);
    const updated = state.regions.find((region) => region.id === edit.id);
    if (updated) selectSavedRegion(updated);
    el("geometry-status").textContent = `Bounds saved as geometry revision ${result.geometry_revision}; previous bounds remain in history.`;
  } catch (error) {
    el("geometry-status").textContent = error.status === 409
      ? "Bounds changed since this annotation was opened. Reopen it before saving."
      : `Could not save corrected bounds: ${error.message}`;
    button.hidden = false;
    el("cancel-region-bounds").hidden = false;
  } finally {
    button.disabled = false;
  }
});

el("cancel-region-bounds").addEventListener("click", () => {
  if (!state.capture || !state.editRegion) return;
  const current = state.regions.find((region) => region.id === state.editRegion.id);
  if (current) selectSavedRegion(current);
});

el("start-split").addEventListener("click", () => {
  const parent = state.regions.find((region) => region.id === state.editRegion?.id);
  if (!parent || parent.lineage_state === "superseded" || !state.selection) return;
  state.splitMode = true;
  state.splitChildren = [];
  state.splitParentSelection = { ...state.selection };
  el("save-split").hidden = false;
  el("save-split").disabled = true;
  el("cancel-split").hidden = false;
  el("start-split").disabled = true;
  el("split-status").textContent = "Draw each child signal's time/frequency bounds on the waterfall. At least two are required.";
});

el("cancel-split").addEventListener("click", () => {
  const current = state.regions.find((region) => region.id === state.editRegion?.id);
  if (current) selectSavedRegion(current);
});

el("save-split").addEventListener("click", async () => {
  const edit = state.editRegion;
  const captureId = state.capture?.id;
  if (!edit || !captureId || state.splitChildren.length < 2) return;
  const button = el("save-split");
  button.disabled = true;
  try {
    const operation = await request(`/api/captures/${captureId}/regions/${edit.id}/split`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_geometry_revision: edit.geometry_revision, children: state.splitChildren }),
    });
    if (state.capture?.id !== captureId) return;
    state.splitMode = false;
    state.regions = await request(`/api/captures/${captureId}/regions`);
    renderRegions();
    await refreshSceneCoverage(captureId);
    const firstChild = state.regions.find((region) => region.id === operation.result_region_ids[0]);
    if (firstChild) selectSavedRegion(firstChild);
    el("split-status").textContent = `Split recorded · ${operation.result_region_ids.length} new regions · parent retained as superseded history.`;
  } catch (error) {
    el("split-status").textContent = error.status === 409
      ? "This region changed or was already superseded. Reopen the current region before splitting."
      : `Split failed: ${error.message}`;
  } finally {
    button.disabled = state.splitChildren.length < 2;
  }
});

el("merge-regions").addEventListener("click", async () => {
  const captureId = state.capture?.id;
  const sources = state.regions.filter((region) => state.mergeRegionIds.has(region.id) && region.lineage_state !== "superseded");
  if (!captureId || sources.length < 2) return;
  const button = el("merge-regions");
  button.disabled = true;
  el("merge-status").textContent = "Recording the merge relation…";
  try {
    const operation = await request(`/api/captures/${captureId}/regions/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_region_ids: sources.map((region) => region.id),
        expected_geometry_revisions: Object.fromEntries(sources.map((region) => [region.id, region.geometry_revision])),
      }),
    });
    if (state.capture?.id !== captureId) return;
    state.mergeRegionIds.clear();
    state.regions = await request(`/api/captures/${captureId}/regions`);
    renderRegions();
    await refreshSceneCoverage(captureId);
    const merged = state.regions.find((region) => region.id === operation.result_region_id);
    if (merged) selectSavedRegion(merged);
    el("merge-status").textContent = "Merge recorded. Source labels remain unchanged; review the new unknown region separately.";
  } catch (error) {
    el("merge-status").textContent = error.status === 409
      ? "A selected region changed or was already superseded. Refresh the list and try again."
      : `Merge failed: ${error.message}`;
  } finally {
    updateMergeControls();
  }
});

el("start-scene-review").addEventListener("click", startSceneCoverageReview);
el("start-cv-coverage-queue").addEventListener("click", startCvCoverageQueue);
el("cancel-coverage").addEventListener("click", () => {
  state.cvCoverageQueue = null;
  exitSceneCoverageReview();
});
el("mark-coverage-uncertain").addEventListener("click", () => beginCoverageMask("uncertain_regions"));
el("mark-coverage-ignore").addEventListener("click", () => beginCoverageMask("ignore_regions"));
el("cancel-coverage-mask").addEventListener("click", cancelCoverageMask);
el("refresh-coverage").addEventListener("click", () => refreshSceneCoverage());
el("save-coverage").addEventListener("click", async () => {
  const captureId = state.capture?.id;
  const selection = state.selection;
  if (!captureId || !selection || state.liveView) return;
  const button = el("save-coverage");
  button.disabled = true;
  el("coverage-status").textContent = "Saving scene review…";
  const payload = {
    sample_start: selection.sample_start,
    sample_count: selection.sample_count,
    frequency_low_hz: state.coverageEdit?.frequencyUnknown ? null : selection.frequency_low_hz,
    frequency_high_hz: state.coverageEdit?.frequencyUnknown ? null : selection.frequency_high_hz,
    task: el("coverage-task").value,
    taxonomy_scope: el("coverage-scope").value.split(",").map((item) => item.trim()).filter(Boolean),
    state: el("coverage-state").value,
    ignore_regions: state.coverageMasks.ignore_regions,
    uncertain_regions: state.coverageMasks.uncertain_regions,
    notes: el("coverage-notes").value.trim(),
  };
  try {
    const edit = state.coverageEdit;
    const url = edit
      ? `/api/captures/${captureId}/coverage/${edit.id}`
      : `/api/captures/${captureId}/coverage`;
    const result = await request(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(edit ? { ...payload, expected_revision: edit.revision } : payload),
    });
    if (state.capture?.id !== captureId) return;
    state.coverageEdit = {
      id: result.id, revision: result.revision,
      frequencyUnknown: result.frequency_low_hz === null,
    };
    state.coverageWindow = { ...state.selection };
    state.coverageMasks = {
      ignore_regions: result.ignore_regions || [],
      uncertain_regions: result.uncertain_regions || [],
    };
    renderSceneCoverageMasks();
    state.coverageRecords = await request(`/api/captures/${captureId}/coverage`);
    renderSceneCoverage();
    const queue = state.cvCoverageQueue;
    if (queue && queue.captureId === captureId) {
      queue.visited.add(selection.sample_start);
      if (result.state === "complete") queue.completeCount += 1;
      else queue.partialCount += 1;
      nextCvReviewWindow();
    } else {
      button.textContent = `Save review revision ${result.revision + 1}`;
      el("coverage-status").textContent = `${result.state} ${result.task} review saved · revision ${result.revision} · ${result.annotation_set_digest.slice(0, 12)}…`;
    }
  } catch (error) {
    if (state.capture?.id !== captureId) return;
    el("coverage-status").textContent = error.status === 409
      ? "This review changed elsewhere. Refresh scene reviews before editing it."
      : `Scene review failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView || !state.selection;
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
    renderRawBitInspector("2fsk", result);
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
    renderRawBitInspector("ook", result);
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
    renderRawBitInspector("mfsk", result);
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
    renderRawBitInspector("bpsk", result);
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
    renderRawBitInspector("qpsk", result);
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
      row.className = region.lineage_state === "superseded" ? "region-row superseded" : "region-row";
      row.dataset.regionId = region.id;
      const main = document.createElement("div");
      main.className = "region-main";
      main.textContent = [region.family, region.method].filter(Boolean).join(" · ") || "Unknown signal";
      const detail = document.createElement("div");
      detail.className = "region-detail";
      const classifierSummary = classifierEvidenceSummary(region.evidence?.classifier_proposal);
      const detectorEvidence = region.evidence?.detector_proposal;
      const detectorSummary = detectorEvidence
        ? `CV ${Math.round(100 * detectorEvidence.confidence)}% · model ${detectorEvidence.model_sha256.slice(0, 8)}${detectorEvidence.supporting_windows?.length > 1 ? ` · seen in ${detectorEvidence.supporting_windows.length} windows` : ""}`
        : null;
      detail.textContent = `${(region.sample_start / state.capture.sample_rate).toFixed(2)} s · ${(region.sample_count / state.capture.sample_rate).toFixed(2)} s duration${region.lineage_state === "superseded" ? " · superseded by split/merge" : ""}${detectorSummary ? ` · ${detectorSummary}` : ""}${classifierSummary ? ` · ${classifierSummary}` : ""}`;
      const left = document.createElement("div");
      left.append(main, detail);
      const protocol = document.createElement("div");
      protocol.className = "region-protocol";
      protocol.textContent = region.protocol ? `${region.protocol}${region.variant ? ` / ${region.variant}` : ""}` : "Protocol unknown";
      const badge = document.createElement("span");
      badge.className = "region-badge";
      const badgeLabels = { synthetic: "SYNTHETIC", human: "HUMAN LABEL", decoder: "DECODER", external: "EXTERNAL", rsid: "RSID", prediction: "PREDICTION", unknown: "UNKNOWN" };
      const identityKnown = Boolean(region.family || region.method || region.protocol || region.variant);
      badge.textContent = region.lineage_state === "superseded"
        ? "SUPERSEDED"
        : region.origin === "detector_proposal"
          ? `CV · ${region.review_state !== "accepted" ? "PENDING REVIEW" : identityKnown ? "HUMAN LABEL" : "PRESENCE REVIEW"}`
          : region.review_state === "accepted" ? (badgeLabels[region.source] || "ACCEPTED") : "PENDING REVIEW";
      row.append(left, protocol, badge);
      const mergeLabel = document.createElement("label");
      mergeLabel.className = "demod-check region-merge-select";
      const mergeCheckbox = document.createElement("input");
      mergeCheckbox.type = "checkbox";
      mergeCheckbox.dataset.regionMergeId = region.id;
      mergeCheckbox.checked = state.mergeRegionIds.has(region.id);
      mergeCheckbox.disabled = state.liveView || region.lineage_state === "superseded";
      mergeCheckbox.setAttribute("aria-label", `Select ${region.variant || region.protocol || region.method || region.family || "unknown region"} for merge`);
      mergeCheckbox.addEventListener("change", () => {
        if (mergeCheckbox.checked) state.mergeRegionIds.add(region.id);
        else state.mergeRegionIds.delete(region.id);
        updateMergeControls();
      });
      mergeLabel.append(mergeCheckbox, document.createTextNode("Merge"));
      row.append(mergeLabel);
      const review = document.createElement("button");
      review.type = "button";
      review.textContent = region.lineage_state === "superseded" ? "History" : "Review";
      review.addEventListener("click", () => selectSavedRegion(region));
      row.append(review);
      list.append(row);
    }
  }
  renderRegionShortcuts();
  updateMergeControls();
  drawOverlays();
}

function updateMergeControls() {
  const selected = [...state.mergeRegionIds].filter((regionId) =>
    state.regions.some((region) => region.id === regionId && region.lineage_state !== "superseded"),
  );
  state.mergeRegionIds = new Set(selected);
  const button = el("merge-regions");
  button.disabled = selected.length < 2 || state.liveView;
  button.textContent = selected.length >= 2 ? `Merge ${selected.length} selected regions` : "Merge selected regions";
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
function updateCarrierFromSelection(selection) {
  const input = el("fldigi-carrier-hz");
  if (!input) return;
  const low = selection?.frequency_low_hz, high = selection?.frequency_high_hz;
  input.value = Number.isFinite(low) && Number.isFinite(high) && high > low
    ? String(Math.round((low + high) / 2))
    : "";
}
function updateDecoderSelectionControls() {
  const validLiveSelection = Boolean(state.liveView && state.liveSessionId && state.selection);
  const validSavedSelection = Boolean(state.capture && state.selection && !state.liveView);
  const validMode = Boolean(el("fldigi-mode")?.value && !el("fldigi-mode").disabled);
  if (el("decode-selection")) {
    el("decode-selection").disabled = !(validSavedSelection || validLiveSelection) || !validMode;
    el("decode-selection").textContent = validLiveSelection ? "Decode live track window" : "Decode selected region";
  }
  if (el("play-selection")) {
    el("play-selection").disabled = !selectedAudioPlayback && !validSavedSelection;
    if (!selectedAudioPlayback) el("play-selection").textContent = "Replay selected region";
  }
  if (el("tag-as-mode")) el("tag-as-mode").disabled = !validSavedSelection || !validMode;
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
  isolatedTrialActive = false;
  watchedTrialTrack = null;
  state.selectedTrack = null;
  state.frozen = false;
  state.decoderTrial = null;
  el("signal-region-shortcuts").replaceChildren();
  el("saved-bit-annotations").replaceChildren();
  state.bitAnnotations = [];
  state.protocolStencils = [];
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
    items.push({region, text: `${region.lineage_state === "superseded" ? "Superseded · " : "Saved · "}${region.variant || region.protocol || region.method || region.family || "Unknown"}`,
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
  stopSelectedAudioPlayback(null);
  const data = state.waterfall;
  if (!data) return;
  state.coverageEdit = null;
  state.coverageOnly = false;
  el("coverage-panel").hidden = true;
  el("coverage-task").value = "detection";
  el("coverage-state").value = "complete";
  el("coverage-scope").value = "";
  el("save-coverage").textContent = "Record reviewed window";
  el("coverage-status").textContent = "";
  clearError();
  state.selection = {sample_start:region.sample_start, sample_count:region.sample_count,
    frequency_low_hz:region.frequency_low_hz ?? 0, frequency_high_hz:region.frequency_high_hz ?? Math.min(3000,data.sample_rate/2)};
  state.coverageEdit = null;
  state.coverageOnly = false;
  el("coverage-panel").hidden = true;
  el("save-coverage").textContent = "Record reviewed window";
  updateSceneReviewAvailability();
  updateCarrierFromSelection(state.selection);
  state.selectedTrack = null;
  state.editRegion = {id:region.id, revision:region.revision, geometry_revision:region.geometry_revision};
  setModeProposalSource(region);
  state.labelModeId = region.mode_id || null;
  el("mode-search").value = region.mode_id || "";
  el("mode-resolution").textContent = region.mode_id ? `Canonical mode ${region.mode_id} · taxonomy retained with registry version in label evidence` : "Partial labels are welcome; unknown attributes stay blank.";
  el("missed-signal").checked = region.origin === "user_added_missed_detection";
  el("feedback-panel").hidden = region.origin !== "detector_proposal" || region.lineage_state === "superseded";
  el("feedback-status").textContent = "";
  el("bounds-panel").hidden = false;
  el("edit-region-bounds").disabled = region.lineage_state === "superseded";
  el("save-region-bounds").hidden = true;
  el("cancel-region-bounds").hidden = true;
  state.adjustingGeometry = false;
  state.geometryDraft = false;
  state.splitMode = false;
  state.splitChildren = [];
  state.splitParentSelection = null;
  el("start-split").disabled = region.lineage_state === "superseded";
  el("save-split").hidden = true;
  el("save-split").disabled = true;
  el("cancel-split").hidden = true;
  el("split-status").textContent = "";
  el("geometry-status").textContent = `Geometry revision ${region.geometry_revision} · original bounds retained when corrected.`;
  reprojectSelection();
  for (const field of ["family","method","protocol","variant","notes"]) el(field).value = region[field] || "";
  el("selection-summary").textContent = region.variant || region.protocol || region.method || region.family || "Saved unknown region";
  el("selection-details").textContent = `${region.source} · ${region.review_state} · ${(region.sample_start/data.sample_rate).toFixed(2)}–${((region.sample_start+region.sample_count)/data.sample_rate).toFixed(2)} s`;
  el("label-state").textContent = `Editing label revision ${region.revision} · original history retained`;
  el("save-region").textContent = "Save revision";
  el("inspector-context").textContent = `Saved annotation · ${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz · revision ${region.revision}`;
  loadLabelHistory(region, state.capture.id);
  loadGeometryHistory(region, state.capture.id);
  loadLineageHistory(region, state.capture.id);
  el("save-region").disabled = state.liveView || region.lineage_state === "superseded";
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
      row.dataset.captureId = capture.id;
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
  if (collapsed) {
    lastExpandedOutputHeight = el("output-dock").getBoundingClientRect().height;
    el("output-dock").style.height = "40px";
    document.documentElement.style.setProperty("--output-dock-height", "40px");
  } else {
    setOutputDockHeight(lastExpandedOutputHeight);
  }
});
{
  const handle = el("output-resize");
  let dragStart = null;
  handle.addEventListener("pointerdown", (event) => {
    if (el("output-dock").classList.contains("collapsed")) return;
    dragStart = { y: event.clientY, height: el("output-dock").getBoundingClientRect().height };
    handle.classList.add("dragging");
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  handle.addEventListener("pointermove", (event) => {
    if (dragStart) setOutputDockHeight(dragStart.height + dragStart.y - event.clientY);
  });
  const finishResize = () => {
    if (!dragStart) return;
    dragStart = null;
    handle.classList.remove("dragging");
    setOutputDockHeight(el("output-dock").getBoundingClientRect().height, true);
  };
  handle.addEventListener("pointerup", finishResize);
  handle.addEventListener("pointercancel", finishResize);
  handle.addEventListener("keydown", (event) => {
    const current = el("output-dock").getBoundingClientRect().height;
    const step = event.shiftKey ? 64 : 24;
    if (event.key === "ArrowUp") setOutputDockHeight(current + step, true);
    else if (event.key === "ArrowDown") setOutputDockHeight(current - step, true);
    else if (event.key === "Home") setOutputDockHeight(90, true);
    else if (event.key === "End") setOutputDockHeight(window.innerHeight * 0.72, true);
    else return;
    event.preventDefault();
  });
  window.addEventListener("resize", () => {
    if (!el("output-dock").classList.contains("collapsed")) setOutputDockHeight(el("output-dock").getBoundingClientRect().height);
  });
}
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
  state.selectedTrack = null;
  el("cv-presence-row").hidden = true;
  el("cv-presence-confirmed").checked = false;
  state.editRegion = null;
  state.coverageEdit = null;
  state.coverageOnly = false;
  state.coverageWindow = null;
  state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
  state.coverageMaskMode = null;
  el("coverage-panel").hidden = true;
  el("coverage-task").value = "detection";
  el("coverage-state").value = "complete";
  el("coverage-scope").value = "";
  el("coverage-notes").value = "";
  el("save-coverage").textContent = "Record reviewed window";
  el("coverage-status").textContent = "";
  el("coverage-mask-reason").value = "";
  el("coverage-mask-status").textContent = "Start a scene review to mark bounded subregions.";
  renderSceneCoverageMasks();
  updateCoverageMaskControls();
  el("save-region").disabled = state.liveView || !state.capture;
  el("save-region").textContent = "Save region label";
  setModeProposalSource(null);
  state.labelModeId = null;
  el("mode-search").value = "";
  el("mode-resolution").textContent = "Choose a known mode to fill its family and protocol, or leave blank for an unknown/partial label.";
  el("missed-signal").checked = false;
  el("feedback-panel").hidden = true;
  el("feedback-status").textContent = "";
  el("bounds-panel").hidden = true;
  state.splitMode = false;
  state.splitChildren = [];
  state.splitParentSelection = null;
  el("save-split").hidden = true;
  el("cancel-split").hidden = true;
  el("split-status").textContent = "";
  el("merge-status").textContent = "";
  el("save-region-bounds").hidden = true;
  el("cancel-region-bounds").hidden = true;
  el("geometry-status").textContent = "";
  state.adjustingGeometry = false;
  state.geometryDraft = false;
  el("save-region").textContent = "Save region label";
  el("label-history").hidden = true;
  el("label-history-list").replaceChildren();
  el("label-state").textContent = "Partial labels are welcome.";
  for(const id of ["family","method","protocol","variant","notes"]) el(id).value = "";
  updateSceneReviewAvailability();
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

async function loadGeometryHistory(region,captureId) {
  const target=el("geometry-history-list");
  target.textContent="Loading bounds revisions…";
  try {
    const history=await request(`/api/captures/${captureId}/regions/${region.id}/geometry`);
    if(state.capture?.id!==captureId || state.editRegion?.id!==region.id) return;
    target.replaceChildren();
    for(const item of history) {
      const line=document.createElement("p");
      const t0=(item.sample_start/state.capture.sample_rate).toFixed(2);
      const t1=((item.sample_start+item.sample_count)/state.capture.sample_rate).toFixed(2);
      const frequency=item.frequency_low_hz===null ? "frequency unknown" : `${item.frequency_low_hz.toFixed(0)}–${item.frequency_high_hz.toFixed(0)} Hz`;
      line.textContent=`Geometry ${item.revision} · ${t0}–${t1} s · ${frequency}`;
      target.append(line);
    }
  } catch(error) {
    if(state.editRegion?.id===region.id) target.textContent=`Bounds history unavailable · ${error.message}`;
  }
}

async function loadLineageHistory(region,captureId) {
  const target=el("lineage-history-list");
  target.textContent="Loading split/merge history…";
  try {
    const history=await request(`/api/captures/${captureId}/regions/${region.id}/lineage`);
    if(state.capture?.id!==captureId || state.editRegion?.id!==region.id) return;
    target.replaceChildren();
    for(const operation of history) {
      const item=document.createElement("p");
      const sources=operation.source_region_ids.map((id)=>id===region.id?"this region":id.slice(0,8)).join(", ");
      const results=operation.result_region_ids.map((id)=>id===region.id?"this region":id.slice(0,8)).join(", ");
      item.textContent=`${operation.relation} · ${sources} → ${results}`;
      target.append(item);
    }
    if(!history.length) target.textContent="No split/merge lineage recorded.";
  } catch(error) {
    if(state.editRegion?.id===region.id) target.textContent=`Lineage history unavailable · ${error.message}`;
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
