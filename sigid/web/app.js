import { liveTimeLabels } from "/waterfall-window.mjs";
import { decodeSpectrumPacket, LiveWaterfall, liveDisplayBounds } from "/live-waterfall.mjs";
import { createDemodExport, downloadDemodExport } from "/demod-export.mjs";
import { bitSelectionRange, bitstreamMetrics, bytesForBits, framingCandidates, transformBitstream, createBitAnnotation } from "/bit-inspector.mjs";
import { narrowKnownModes } from "/mode-candidates.mjs";
import { layoutSignalLabels, signalCaption, hitSignal } from "/signal-overlays.mjs";

const el = (id) => document.getElementById(id);
const state = { capture: null, waterfall: null, drag: null, resizeDrag: null, moveDrag: null, selection: null, regions: [], tracks: [], cvDetections: [], cvDetectionCache: new Map(), cvDismissedCount: 0, cvRejectedProposals: [], cvAvailable: false, cvSimilarAvailable: false, bitAnnotations: [], protocolStencils: [], liveView: false, liveSessionId: null, modelAvailable: false, modelTraining: false, view: null, demodExports: {}, selectedTrack: null, classifierProposal: null, frozen: false, labelModeId: null, modeSearchResults: [], modeSearchRequest: 0, applyingMode: false, adjustingGeometry: false, geometryDraft: false, adjustingCandidate: null, drawingSignalBox: false, batchBoxMode: false, batchBoxes: [], batchBoxUndo: [], batchReviewBox: null, batchSaveKey: null, batchSaveDigest: null, batchSaving: false, windowRegionEdits: new Map(), windowRegionDeletes: new Map(), windowMerges: [], windowIdentityLinks: [], identityLinkSelection: new Set(), splitMode: false, splitChildren: [], splitParentSelection: null, mergeRegionIds: new Set(), selectionMergeRegions: [], excludedMergeCandidates: new Set(), modeProposalRegion: null, proposalCaptureRequest: 0, proposalEvidenceCaptures: [] };
state.geometryMergeRegions = [];
state.showSupersededRegions = false;
let nativeDecoderBindings = [];
function savedInspectorPreference(key, accepted, fallback) {
  try {
    const value = localStorage.getItem(key);
    return accepted.includes(value) ? value : fallback;
  } catch {
    return fallback;
  }
}
const FREQUENCY_NOTCH_KEY = "sig-id-frequency-notches";
const NOTCH_COLORS = ["#ff6b6b", "#ffd166", "#65d6a6", "#68b7ff", "#d58cff", "#ff9f5a"];
function loadFrequencyNotches() {
  try {
    const values = JSON.parse(localStorage.getItem(FREQUENCY_NOTCH_KEY) || "[]");
    return Array.isArray(values) ? values.filter((item) => Number.isFinite(item.low_hz)
      && Number.isFinite(item.high_hz) && item.high_hz > item.low_hz) : [];
  } catch { return []; }
}
state.frequencyNotches = loadFrequencyNotches();
state.notchPlacement = false;
state.notchDrag = null;
function frequencyNotchMatches(candidate) {
  const low = Number(candidate?.frequency_low_hz);
  const high = Number(candidate?.frequency_high_hz);
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return false;
  const center = (low + high) / 2;
  return state.frequencyNotches.some((notch) => {
    const overlap = Math.max(0, Math.min(high, notch.high_hz) - Math.max(low, notch.low_hz));
    return (center >= notch.low_hz && center <= notch.high_hz) || overlap / (high - low) >= 0.5;
  });
}
function withoutNotchedCandidates(items) {
  return items.filter((item) => !frequencyNotchMatches(item));
}
function visibleCvDetections() { return withoutNotchedCandidates(state.cvDetections); }
function visibleTracks() { return withoutNotchedCandidates(state.tracks); }
function saveFrequencyNotches() {
  try { localStorage.setItem(FREQUENCY_NOTCH_KEY, JSON.stringify(state.frequencyNotches)); } catch {}
  if (state.selectedTrack && !state.editRegion && !state.batchReviewBox
      && frequencyNotchMatches(state.selectedTrack)) {
    state.selectedTrack = null;
    state.selection = null;
    state.classifierProposal = null;
    beginNewRegion();
  }
  renderFrequencyNotches();
  renderCvDetections();
  renderTracks();
  drawOverlays();
}
state.inspectorMode = savedInspectorPreference("sig-id-inspector-mode", ["classify", "annotate"], "classify");
state.inspectorTab = savedInspectorPreference("sig-id-inspector-tab", ["label", "decode", "raw", "signals"], "label");
state.inspectorDetailLevel = savedInspectorPreference("sig-id-inspector-detail", ["simple", "advanced"], "simple");
const classifierCropOptions = Array.from({ length: 20 }, (_, index) => String(index + 1));
const classifierCropCount = savedInspectorPreference("sig-id-classifier-crop-count", classifierCropOptions, "1");
el("classifier-crop-count").value = classifierCropCount;
el("live-cv-enabled").checked = savedInspectorPreference("sig-id-live-cv-enabled", ["on", "off"], "on") === "on";
el("live-classifier-enabled").checked = savedInspectorPreference("sig-id-live-classifier-enabled", ["on", "off"], "on") === "on";
for (const id of ["live-cv-enabled", "live-classifier-enabled"]) {
  el(id).addEventListener("change", () => {
    try { localStorage.setItem(`sig-id-${id}`, el(id).checked ? "on" : "off"); } catch {}
    if (!el("live-cv-enabled").checked) {
      state.cvDetections = [];
      renderCvDetections();
    }
    if (el("live-cv-enabled").checked && !el("live-classifier-enabled").checked) {
      for (const detection of state.cvDetections) delete detection.classifier_proposal;
      renderCvDetections();
    }
    state.liveTrackModeMessage = el("live-cv-enabled").checked
      ? "CV boxes are active above; heuristic track boxes are hidden. Turn CV off to inspect classifier-only track candidates."
      : el("live-classifier-enabled").checked
        ? null
        : "Candidate analysis is paused. Turn on CV signal boxes or classifier suggestions to show candidates.";
    if (el("live-cv-enabled").checked || !el("live-classifier-enabled").checked) {
      state.tracks = [];
      renderTracks();
    }
    if (state.liveView && !state.frozen) drawWaterfall();
  });
}
const liveCvMergeOptions = Array.from({ length: 15 }, (_, index) => String((index + 2) * 5));
const liveCvMergeOverlap = savedInspectorPreference("sig-id-live-cv-merge-overlap", liveCvMergeOptions, "30");
el("live-cv-merge-overlap").value = liveCvMergeOverlap;
const ANNOTATION_WINDOW_SECONDS = 10;
state.annotationWindowIndex = 0;
function annotationWindowBounds(capture = state.capture) {
  if (!capture) return {startSample: 0, sampleCount: 0, endSample: 0, index: 0, total: 0};
  const windowLength = Math.max(1, Math.round(capture.sample_rate * ANNOTATION_WINDOW_SECONDS));
  const total = Math.max(1, Math.ceil(capture.sample_count / windowLength));
  state.annotationWindowIndex = Math.max(0, Math.min(total - 1, state.annotationWindowIndex || 0));
  const startSample = state.annotationWindowIndex * windowLength;
  const sampleCount = Math.max(0, Math.min(windowLength, capture.sample_count - startSample));
  return {startSample, sampleCount, endSample: startSample + sampleCount, index: state.annotationWindowIndex, total};
}
function updateAnnotationWindowView(redraw = true) {
  const bounds = annotationWindowBounds();
  if (state.waterfall && !state.liveView && state.view && bounds.sampleCount > 0) {
    state.view.timeStartSample = bounds.startSample;
    state.view.timeEndSample = bounds.endSample;
    state.view.followTime = false;
    reprojectSelection();
    if (redraw) drawWaterfall();
  }
  return bounds;
}
function setRegionSaveDisabled(disabled) {
  const value = Boolean(disabled);
  el("save-region").disabled = value;
  el("save-annotations").disabled = value;
  el("save-annotations-close").disabled = value;
  el("save-annotations-complete").disabled = value;
}

function syncLiveAnnotationSaveButton() {
  const selection = state.selection;
  const windowStart = state.liveWindowStartSample;
  const windowEnd = state.liveWindowEndSample;
  const valid = Boolean(state.liveView && state.liveSessionId && selection
    && Number.isFinite(windowStart) && Number.isFinite(windowEnd)
    && selection.sample_start >= windowStart
    && selection.sample_start + selection.sample_count <= windowEnd
    && windowEnd - windowStart >= (state.liveSampleRate || 0) * 10);
  el("save-annotations").disabled = !valid;
  el("save-annotations").textContent = "Save annotations";
  el("save-annotations").title = valid
    ? "Save this labeled signal as a training candidate from the current 10-second live window."
    : "Select a signal fully inside a complete 10-second live window to save a training candidate.";
}

function recordCompleteReviewWindowCoverage(capture, window = annotationWindowBounds(capture)) {
  return request(`/api/captures/${capture.id}/coverage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sample_start: window.startSample,
      sample_count: window.sampleCount,
      frequency_low_hz: 0,
      frequency_high_hz: state.cvReviewMaxFrequencyHz,
      task: "detection",
      taxonomy_scope: [],
      state: "complete",
      ignore_regions: [],
      uncertain_regions: [],
      notes: "",
    }),
  });
}
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
      state.fldigiModes = [];
      el("fldigi-select").disabled = true;
      el("fldigi-check-wefax").disabled = true;
      updateDecoderSelectionControls();
      return false;
    }
    label.textContent = `${status.name} ${status.version} · ${status.current_mode} · ${status.mode_count} modes · audio route not verified`;
    el("connection").lastChild.textContent = ` ${status.name} receive connected`;
    if (fldigiOffset === null) fldigiOffset = status.receive_text_length;
    const modes = await request("/api/fldigi/modes");
    state.fldigiModes = modes;
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

async function refreshNativeDecoderStatus() {
  const statusNode = el("native-decoder-status");
  const resultNode = el("native-decoder-results");
  if (!statusNode || !resultNode) return;
  try {
    const response = await request("/api/decoder-backends");
    const backends = response.backends || [];
    statusNode.textContent = backends.map((backend) => {
      const modeCount = (backend.modes || []).length;
      return `${backend.application || backend.backend_id}: ${backend.connected ? "ready" : "not connected"} · ${modeCount} modes · ${backend.note || ""}`;
    }).join("\n") || "No native decoder backends are registered.";
    statusNode.style.whiteSpace = "pre-line";
    if (response.plugin_errors?.length) {
      statusNode.textContent += `\nPlugin load errors: ${response.plugin_errors.map((item) => `${item.plugin}: ${item.error}`).join("; ")}`;
    }
    nativeDecoderBindings = backends.flatMap((backend) => (backend.modes || [])
      .filter((mode) => backend.connected && mode.selectable)
      .map((mode) => ({...mode, backend_id: backend.backend_id, application: backend.application})));
    const modeSelect = el("native-decoder-mode");
    const selectedValue = modeSelect.value;
    modeSelect.replaceChildren(...nativeDecoderBindings.map((mode) => new Option(
      `${mode.application} · ${mode.display_name}`, `${mode.backend_id}|${mode.mode_id}`,
    )));
    modeSelect.disabled = nativeDecoderBindings.length === 0;
    if (nativeDecoderBindings.some((mode) => `${mode.backend_id}|${mode.mode_id}` === selectedValue)) {
      modeSelect.value = selectedValue;
    } else {
      const selectedLabelMode = nativeDecoderBindings.find((mode) => mode.mode_id === state.labelModeId);
      if (selectedLabelMode) modeSelect.value = `${selectedLabelMode.backend_id}|${selectedLabelMode.mode_id}`;
    }
    updateDecoderSelectionControls();
    const events = [];
    for (const backend of backends) {
      if (backend.backend_id !== "wsjtx" || !backend.connected) continue;
      const received = await request(`/api/decoder-backends/${encodeURIComponent(backend.backend_id)}/events?limit=100`);
      events.push(...received);
    }
    if (!events.length && !resultNode.children.length) {
      resultNode.replaceChildren(Object.assign(document.createElement("p"), {
        className: "muted",
        textContent: "No new native decoder results. Results from WSJT-X appear here when its UDP reporting is configured.",
      }));
      return;
    }
    if (!events.length) return;
    for (const event of events) {
      const row = document.createElement("div");
      row.className = "decode-run";
      const products = (event.products || []).map((product) => product.kind === "text" ? product.content : `[${product.kind} product]`);
      const frequency = Number.isFinite(event.frequency_hz) ? ` · ${event.frequency_hz} Hz audio offset` : "";
      const snr = Number.isFinite(event.snr_db) ? ` · ${event.snr_db} dB SNR` : "";
      row.textContent = `${event.native_mode} · ${event.observed_utc}${frequency}${snr}\n${products.join("\n") || "No text product"}`;
      row.style.whiteSpace = "pre-line";
      resultNode.prepend(row);
    }
    while (resultNode.children.length > 100) resultNode.lastElementChild.remove();
    updateDecoderSelectionControls();
  } catch (error) {
    statusNode.textContent = `Native decoder status unavailable · ${error.message}`;
  }
}

el("native-decoder-mode").addEventListener("change", updateDecoderSelectionControls);
el("native-decode-selection").addEventListener("click", async () => {
  const selection = state.selection;
  const capture = state.capture;
  const [backendId, modeId] = el("native-decoder-mode").value.split("|");
  if (!capture || state.liveView || !selection || !backendId || !modeId) return;
  const linkedRegion = state.editRegion && state.regions.find((region) =>
    region.id === state.editRegion.id && region.lineage_state === "active"
      && region.sample_start === selection.sample_start && region.sample_count === selection.sample_count,
  );
  const button = el("native-decode-selection");
  button.disabled = true;
  el("native-decode-status").textContent = `Sending selected audio to ${backendId} · ${modeId}…`;
  try {
    const result = await request(`/api/captures/${encodeURIComponent(capture.id)}/decoder-backends/${encodeURIComponent(backendId)}/decode`, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({mode_id: modeId, sample_start: selection.sample_start,
        sample_count: selection.sample_count, region_id: linkedRegion?.id || null,
        carrier_hz: Number.isFinite(selection.frequency_low_hz) && Number.isFinite(selection.frequency_high_hz)
          ? (selection.frequency_low_hz + selection.frequency_high_hz) / 2 : null}),
    });
    const text = (result.products || []).filter((product) => product.kind === "text")
      .map((product) => product.content).join("\n");
    el("native-decode-status").textContent = `${result.status.toUpperCase()} · ${result.application} · ${result.note}`;
    showOutput("text");
    state.textOutputContext = `${result.application} · ${result.native_mode} · exact selected window`;
    el("output-context").textContent = state.textOutputContext;
    el("decoded-text").textContent = text || "No decoded text in this window.";
    state.decodeRuns = await request(`/api/captures/${encodeURIComponent(capture.id)}/decode-runs`);
    renderDecodeRuns();
  } catch (error) {
    el("native-decode-status").textContent = `Native decode failed · ${error.message}`;
  } finally {
    updateDecoderSelectionControls();
  }
});

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
        detector_evidence: !liveSessionId && state.selectedTrack?.cv_proposal_id
          ? state.selectedTrack.detector_evidence || null : null,
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

el("auto-identify-selection").addEventListener("click", async () => {
  const selection = state.selection;
  const captureId = state.capture?.id;
  const liveSessionId = state.liveView ? state.liveSessionId : null;
  const available = new Set(state.fldigiModes || []);
  if (!selection || (!captureId && !liveSessionId) || !available.size) return;
  const carrierText = el("fldigi-carrier-hz").value.trim();
  const carrierHz = carrierText ? Number(carrierText) : null;
  if (carrierText && (!Number.isFinite(carrierHz) || carrierHz <= 0 || carrierHz > 20000)) {
    el("replay-status").textContent = "Enter an audio carrier frequency from 1 to 20000 Hz, or clear the field to leave it unset.";
    return;
  }
  const proposal = state.classifierProposal?.selection === selection
    && (!captureId || state.classifierProposal.capture_id === captureId)
    ? state.classifierProposal.result : null;
  let candidates = [];
  const searchTerm = proposal?.method || proposal?.family;
  if (searchTerm) {
    try {
      const response = await request(`/api/modes?q=${encodeURIComponent(searchTerm)}&limit=50`);
      const expected = (proposal.method || proposal.family).trim().toLocaleLowerCase();
      candidates = response.modes
        .filter((item) => (proposal.method
          ? item.method?.trim().toLocaleLowerCase() === expected
          : item.family?.trim().toLocaleLowerCase() === expected))
        .map((item) => item.display_name)
        .filter((name) => available.has(name));
    } catch { /* use the known tested receive candidates below */ }
  }
  for (const fallback of ["BPSK31", "RTTY", "THOR16", "MFSK16", "QPSK31", "OLIVIA-8/250"]) {
    if (available.has(fallback) && !candidates.includes(fallback)) candidates.push(fallback);
  }
  candidates = candidates.slice(0, 6);
  if (!candidates.length) {
    el("replay-status").textContent = "No candidate receive modes are available in the connected Fldigi catalog.";
    return;
  }
  const button = el("auto-identify-selection");
  button.disabled = true;
  el("replay-status").textContent = `Checking RSID, then testing ${candidates.join(", ")} against this exact audio window…`;
  try {
    const endpoint = liveSessionId
      ? `/api/live/sessions/${encodeURIComponent(liveSessionId)}/decode`
      : `/api/captures/${captureId}/decode`;
    const result = await request(endpoint, {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        mode: candidates[0], auto_identify: true, candidate_modes: candidates,
        sample_start: selection.sample_start, sample_count: selection.sample_count,
        track_id: state.selectedTrack?.track_id || null, carrier_hz: carrierHz,
        region_id: null,
      }),
    });
    el("replay-status").textContent = `Auto-identification ${result.status} · ${candidates.length} candidates · isolated ${liveSessionId ? "live track" : "selected region"} input · tentative only`;
    if (liveSessionId) pollLiveDecodeRun(liveSessionId, result.run_id);
    else pollSavedDecodeRun(captureId, result.run_id);
  } catch (error) {
    el("replay-status").textContent = `Auto-identification could not start · ${error.message}`;
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
    const featureScope = model.feature_frequency_scope === "full_audio_passband"
      ? "full passband features"
      : model.feature_frequency_scope === "roi_bandpass_expanded"
        ? "box ROI features"
        : model.feature_frequency_scope === "full_range_roi_bandpass"
          ? "full-range ROI features"
          : `${model.feature_frequency_scope} features`;
    const classifierWindow = model.classifier_window_seconds
      ? ` · ${Number(model.classifier_window_seconds).toFixed(1)} s per crop`
      : "";
    el("model-status").textContent = model.available
      ? `Method model ready · method ${(100 * model.threshold).toFixed(0)}% · family ${(100 * model.family_threshold).toFixed(0)}%${unknownStatus} · ${featureScope}${classifierWindow} · synthetic only`
      : `No model loaded${model.error ? ` · ${model.error}` : " · configure SIG_ID_MODEL_PATH"}`;
    updateModelButton();
    updateTrainingButton();
    renderBatchBoxes();
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
      ? `YOLO detector ready · ${backendDescription}`
      : status.error
        ? `Optional CV detector unavailable · ${status.error}`
        : `No CV detector loaded · checked ${status.active_models_file || "active model configuration"}`;
    el("cv-model-details").textContent = status.available
      ? `${evaluationDescription} · train sources: ${trainingKinds} · held-out sources: ${testKinds} · ${status.training_scope || "training scope not recorded"} · ${status.window_seconds}s windows · confidence ${(100 * status.confidence_threshold).toFixed(0)}% · class agnostic`
      : "Install the cv extra and configure SIG_ID_CV_MODEL_PATH to enable detection.";
    updateCvButton();
    updateTrainingButton();
    updateCvSimilarButton();
  } catch (error) {
    state.cvAvailable = false;
    state.cvSimilarAvailable = false;
    el("cv-status").textContent = `CV status unavailable · ${error.message}`;
    el("cv-model-details").textContent = "Detector information is unavailable.";
    updateCvButton();
    updateCvSimilarButton();
  }
}
refreshCvStatus();

function updateTrainingButton() {
  const button = el("train-models");
  if (!button) return;
  button.disabled = state.modelTraining || !state.modelAvailable || !state.cvAvailable;
  button.textContent = state.modelTraining ? "Training models…" : "Train both models from new reviews";
}

async function refreshTrainingStatus() {
  try {
    const job = await request("/api/models/training/status");
    state.modelTraining = job.status === "running";
    const status = el("train-models-status");
    const progress = el("model-training-progress");
    const percent = Math.max(0, Math.min(100, Number(job.progress) || 0));
    progress.value = percent;
    progress.hidden = !state.modelTraining && job.status !== "complete";
    if (job.status === "running") {
      status.textContent = `${job.stage || "Training both models…"} · ${percent}%`;
      window.setTimeout(refreshTrainingStatus, 2500);
    } else if (job.status === "failed") {
      status.textContent = `Training failed · ${job.error || "see server log"}`;
    } else if (job.status === "complete") {
      const examples = job.result?.new_classifier_examples ?? 0;
      status.textContent = `Both models updated · ${examples} new classifier crops · active until replaced by a base-model pipeline run`;
      refreshModelStatus();
      refreshCvStatus();
    } else {
      status.textContent = "Uses fresh, fully reviewed live captures; base training remains in the full pipeline.";
    }
    updateTrainingButton();
  } catch (error) {
    el("train-models-status").textContent = `Training status unavailable · ${error.message}`;
  }
}
refreshTrainingStatus();

el("train-models").addEventListener("click", async () => {
  const button = el("train-models");
  button.disabled = true;
  el("train-models-status").textContent = "Starting paired model training…";
  el("model-training-progress").value = 0;
  el("model-training-progress").hidden = false;
  try {
    await request("/api/models/train", { method: "POST" });
    state.modelTraining = true;
    updateTrainingButton();
    refreshTrainingStatus();
  } catch (error) {
    el("train-models-status").textContent = `Could not start training · ${error.message}`;
    updateTrainingButton();
  }
});

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
    const staleCount = Number(result.stale_index_entries_skipped || 0);
    const staleNote = staleCount
      ? ` · ${staleCount} stale reference${staleCount === 1 ? " was" : "s were"} skipped; rebuild the index to refresh it`
      : "";
    el("cv-similar-status").textContent = result.count
      ? `Nearest ${result.count} saved patch${result.count === 1 ? "" : "es"} · similarity is not an identification${staleNote}`
      : staleCount
        ? `No other current references found · ${staleCount} stale reference${staleCount === 1 ? " was" : "s were"} skipped; rebuild the index to refresh it`
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

function applyCvDetectionResult(capture, result, { reused = false } = {}) {
  const dismissed = result.dismissed_proposals || [];
  const dismissedIds = new Set(dismissed.map((item) => item.proposal_id));
  const detections = (result.detections || [])
    .map((detection) => ({
      ...detection,
      cv_proposal_id: detection.proposal_id || detection.cv_proposal_id,
      detector_evidence: detection.detector_evidence || detectorEvidenceFor(detection),
    }))
    .filter((detection) => !dismissedIds.has(detection.cv_proposal_id));
  state.cvDetections = detections;
  state.cvRejectedProposals = dismissed;
  state.cvDismissedCount = 0;
  state.cvDetectionCache.set(capture.id, {
    model_sha256: result.model_sha256,
    detections,
    dismissed_proposals: dismissed,
    total_proposed: result.detections?.length || 0,
  });
  renderCvDetections();
  renderCvDismissedProposals();
  const visibleCount = visibleCvDetections().length;
  const classifierCount = visibleCvDetections().filter((item) => item.classifier_proposal?.available).length;
  el("cv-status").textContent = reused
    ? `Showing saved CV scan · ${visibleCount} boxes visible${detections.length > visibleCount ? ` · ${detections.length - visibleCount} suppressed by frequency notches` : ""} · ${dismissed.length} saved dismissals`
    : `CV proposed ${result.detections.length} box${result.detections.length === 1 ? "" : "es"} · ${visibleCount} visible${detections.length > visibleCount ? ` · ${detections.length - visibleCount} suppressed by frequency notches` : ""} · ${dismissed.length} saved dismissals · ${result.classifier_available ? `classifier scored ${classifierCount}` : "classifier unavailable"} · provisional; review before labeling or decoding`;
  drawOverlays();
}

function mergeLiveCvDetections(incoming) {
  const retained = [...state.cvDetections];
  const minimumOverlap = Number(el("live-cv-merge-overlap").value) / 100;
  const maxScanGapSamples = Math.max(1, state.liveSampleRate || 1) * 3;
  const overlapFraction = (leftStart, leftEnd, rightStart, rightEnd) => {
    const intersection = Math.max(0, Math.min(leftEnd, rightEnd) - Math.max(leftStart, rightStart));
    return intersection / Math.max(1, Math.min(leftEnd - leftStart, rightEnd - rightStart));
  };
  for (const detection of incoming || []) {
    const detectionEnd = detection.sample_start + detection.sample_count;
    const match = retained.find((existing) => {
      const existingEnd = existing.sample_start + existing.sample_count;
      const timeOverlap = overlapFraction(existing.sample_start, existingEnd,
        detection.sample_start, detectionEnd) >= minimumOverlap;
      const timeGap = Math.max(0, Math.max(existing.sample_start, detection.sample_start)
        - Math.min(existingEnd, detectionEnd));
      const sameFrequency = overlapFraction(existing.frequency_low_hz, existing.frequency_high_hz,
        detection.frequency_low_hz, detection.frequency_high_hz) >= minimumOverlap;
      return sameFrequency && (timeOverlap || timeGap <= maxScanGapSamples);
    });
    if (!match) {
      retained.push(detection);
      continue;
    }
    // Join the observed bounds so a signal found in several rolling windows
    // gets one continuous box instead of a short box from its first sighting.
    const sampleEnd = Math.max(match.sample_start + match.sample_count,
      detection.sample_start + detection.sample_count);
    match.sample_start = Math.min(match.sample_start, detection.sample_start);
    match.sample_count = sampleEnd - match.sample_start;
    match.frequency_low_hz = Math.min(match.frequency_low_hz, detection.frequency_low_hz);
    match.frequency_high_hz = Math.max(match.frequency_high_hz, detection.frequency_high_hz);
    match.confidence = Math.max(match.confidence || 0, detection.confidence || 0);
    if (detection.classifier_proposal?.available
      && (!match.classifier_proposal?.available
        || (detection.classifier_proposal.confidence || 0) > (match.classifier_proposal.confidence || 0))) {
      match.classifier_proposal = detection.classifier_proposal;
    }
    const windows = new Set([...(match.supporting_windows || []), ...(detection.supporting_windows || [])]);
    if (windows.size) match.supporting_windows = [...windows];
  }
  state.cvDetections = retained.sort((left, right) => left.sample_start - right.sample_start);
}

function restoreCachedCvDetection(capture) {
  const cached = state.cvDetectionCache.get(capture.id);
  if (!cached) return false;
  applyCvDetectionResult(capture, {
    model_sha256: cached.model_sha256,
    detections: cached.detections,
    dismissed_proposals: cached.dismissed_proposals,
  }, { reused: true });
  return true;
}

function removeSavedCvProposal(proposalId) {
  if (!proposalId) return;
  const previousCount = state.cvDetections.length;
  state.cvDetections = state.cvDetections.filter((detection) => detection.cv_proposal_id !== proposalId);
  if (state.cvDetections.length === previousCount) return;
  const cached = state.cvDetectionCache.get(state.capture?.id);
  if (cached) state.cvDetectionCache.set(state.capture.id, { ...cached, detections: state.cvDetections });
  renderCvDetections();
  drawOverlays();
}

function cvClassifierCaption(detection) {
  const proposal = detection?.classifier_proposal;
  if (!proposal) return "classifier not run";
  if (!proposal.available) return `classifier skipped · ${proposal.reason || "unavailable"}`;
  const candidateFamily = proposal.candidate_family || proposal.family;
  const candidateMode = proposal.candidate_mode_name || proposal.mode_name || proposal.variant || proposal.method;
  const candidate = proposal.abstained
    ? [candidateFamily, candidateMode].filter(Boolean).join(" → ") || proposal.top_candidate || "unknown"
    : proposal.mode_name || proposal.variant || proposal.protocol || proposal.method || proposal.family || "unknown";
  if (proposal.abstained) return `classifier unsure · ${candidate}`;
  return `classifier · ${candidate} · ${(100 * (proposal.confidence || 0)).toFixed(0)}%`;
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
    applyCvDetectionResult(capture, result);
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
  const detections = visibleCvDetections();
  for (const [index, detection] of detections.entries()) {
    const row = document.createElement("div");
    const pendingMerge = pendingWindowMergeSources().proposals.has(detection.cv_proposal_id || detection.proposal_id);
    row.className = `track-row${pendingMerge ? " pending-merge" : ""}`;
    const summary = document.createElement("div");
    summary.className = "track-summary";
    const supportingWindows = (detection.supporting_windows || []).length;
    const corroboration = supportingWindows > 1 ? ` · seen in ${supportingWindows} windows` : "";
    const classification = detection.classifier_proposal
      ? cvClassifierCaption(detection)
      : "identity unknown";
    summary.textContent = `CV box ${index + 1} · ${(detection.confidence * 100).toFixed(1)}% detector score${corroboration} · ${pendingMerge ? "merge staged" : classification}`;
    const detail = document.createElement("div");
    detail.className = "region-detail";
    detail.textContent = `${detection.frequency_low_hz.toFixed(0)}–${detection.frequency_high_hz.toFixed(0)} Hz · ${(detection.sample_start / detection.sample_rate).toFixed(2)}–${((detection.sample_start + detection.sample_count) / detection.sample_rate).toFixed(2)} s`;
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = "Review selection";
    button.addEventListener("click", () => selectCvDetection(detection));
    const adjust = document.createElement("button");
    adjust.type = "button";
    adjust.textContent = "Redraw bounds";
    adjust.title = "Redraw the corrected time/frequency bounds on the waterfall; you can extend beyond the detector box.";
    adjust.addEventListener("click", () => armCandidateRedraw(detection));
    const reject = document.createElement("button");
    reject.type = "button";
    reject.textContent = "Reject box";
    reject.title = "Remove this proposal from the current scan; this does not create a training label.";
    reject.addEventListener("click", () => rejectCvDetection(detection, reject));
    button.disabled = pendingMerge;
    adjust.disabled = pendingMerge;
    reject.disabled = pendingMerge;
    if (state.liveView) {
      summary.textContent = `Live CV box ${index + 1} · ${(detection.confidence * 100).toFixed(1)}% detector score · ${cvClassifierCaption(detection)} · provisional`;
      row.append(summary, detail);
    } else {
      row.append(summary, detail, button, adjust, reject);
    }
    list.append(row);
  }
  if (!detections.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = state.liveView && !el("live-cv-enabled").checked
      ? "CV is paused."
      : state.cvDismissedCount
        ? `No proposals remain · ${state.cvDismissedCount} rejected from this scan; dismissals are not training labels.`
        : state.cvDetections.length
          ? `All ${state.cvDetections.length} detector box${state.cvDetections.length === 1 ? " is" : "es are"} suppressed by frequency notches.`
          : "No boxes proposed. The detector may miss weak or unfamiliar signals.";
    list.append(empty);
  }
}

async function rejectCvDetection(detection, button = null) {
  const capture = state.capture;
  if (!capture || state.liveView) return;
  if (button) {
    button.disabled = true;
    button.textContent = "Saving dismissal…";
  }
  try {
    const saved = await saveCvProposalDecision(capture.id, detection, "dismissed");
    if (state.capture !== capture) return;
    state.cvDetections = state.cvDetections.filter((item) => item !== detection);
    state.cvRejectedProposals = [saved, ...state.cvRejectedProposals.filter((item) => item.proposal_id !== saved.proposal_id)];
    state.cvDismissedCount += 1;
    const cached = state.cvDetectionCache.get(capture.id);
    if (cached) state.cvDetectionCache.set(capture.id, {
      ...cached,
      detections: state.cvDetections,
      dismissed_proposals: state.cvRejectedProposals,
    });
    if (state.selectedTrack?.cv_proposal_id === detection.cv_proposal_id) {
      state.selection = null;
      beginNewRegion();
      el("selection-summary").textContent = "No selection";
      el("selection-details").textContent = "The dismissed CV proposal was cleared from the label editor.";
      el("inspector-context").textContent = "Select a signal region to inspect or label it.";
    }
    renderCvDetections();
    renderCvDismissedProposals();
    drawOverlays();
    el("cv-status").textContent = `${state.cvDetections.length} proposals remain · ${state.cvDismissedCount} dismissed from this scan and saved as review history; no training label was recorded`;
  } catch (error) {
    if (state.capture !== capture) return;
    if (button) {
      button.disabled = false;
      button.textContent = "Reject box";
    }
    el("cv-status").textContent = `Could not save proposal dismissal · ${error.message}`;
  }
}

function detectorEvidenceFor(detection) {
  return {
    detector: "yolo",
    proposal_id: detection.proposal_id || detection.cv_proposal_id || null,
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
  };
}

function savePendingClassifierRegion(capture, selection, track = null, result = null) {
  return request(`/api/captures/${capture.id}/regions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sample_start: selection.sample_start,
      sample_count: selection.sample_count,
      frequency_low_hz: selection.frequency_low_hz,
      frequency_high_hz: selection.frequency_high_hz,
      origin: track?.detector_evidence ? "detector_proposal" : "manual",
      detector_evidence: track?.detector_evidence || null,
      classifier_proposal: {
        sample_start: selection.sample_start,
        sample_count: Math.min(1_000_000, selection.sample_count),
        selection_sample_count: selection.sample_count,
        frequency_low_hz: selection.frequency_low_hz,
        frequency_high_hz: selection.frequency_high_hz,
      },
      prediction_only: true,
      notes: result && !result.family && !result.method
        ? "Classifier abstained; signal identity needs operator review."
        : "Classifier suggestion awaiting operator review.",
    }),
  });
}

async function saveCvProposalDecision(captureId, proposal, decision) {
  const bounds = proposal._originalProposalBounds || proposal;
  return request(`/api/captures/${encodeURIComponent(captureId)}/cv-proposal-decisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      proposal_id: proposal.cv_proposal_id || proposal.proposal_id,
      decision,
      sample_start: bounds.sample_start,
      sample_count: bounds.sample_count,
      frequency_low_hz: bounds.frequency_low_hz,
      frequency_high_hz: bounds.frequency_high_hz,
      confidence: proposal.confidence,
      detector_evidence: proposal.detector_evidence || detectorEvidenceFor(proposal),
      note: "Operator dismissed this proposal; this is not a no-signal annotation.",
    }),
  });
}

function detectionFromCvDecision(decision) {
  const evidence = decision.detector_evidence;
  return {
    sample_start: decision.sample_start,
    sample_count: decision.sample_count,
    frequency_low_hz: decision.frequency_low_hz,
    frequency_high_hz: decision.frequency_high_hz,
    confidence: decision.confidence,
    model_sha256: decision.model_sha256,
    model_version: `sha256:${decision.model_sha256}`,
    box_xyxy_normalized: evidence.box_xyxy_normalized,
    image_width: evidence.image_width,
    image_height: evidence.image_height,
    preprocessing_version: evidence.preprocessing_version,
    window_sample_start: evidence.window_sample_start,
    window_sample_count: evidence.window_sample_count,
    sample_rate: evidence.sample_rate,
    supporting_windows: evidence.supporting_windows || [],
    cv_proposal_id: decision.proposal_id,
    detector_evidence: evidence,
  };
}

function renderCvDismissedProposals() {
  const details = el("cv-dismissed-details");
  const list = el("cv-dismissed-list");
  const proposals = state.cvRejectedProposals || [];
  details.hidden = !proposals.length;
  el("cv-dismissed-summary").textContent = `Saved dismissed CV proposals · ${proposals.length}`;
  list.replaceChildren();
  for (const decision of proposals) {
    const row = document.createElement("div");
    row.className = "track-row";
    const summary = document.createElement("div");
    summary.className = "track-summary";
    summary.textContent = `Dismissed proposal · ${(decision.confidence * 100).toFixed(1)}% detector score · review feedback only`;
    const detail = document.createElement("div");
    detail.className = "region-detail";
    detail.textContent = `${decision.frequency_low_hz.toFixed(0)}–${decision.frequency_high_hz.toFixed(0)} Hz · ${(decision.sample_start / decision.detector_evidence.sample_rate).toFixed(2)}–${((decision.sample_start + decision.sample_count) / decision.detector_evidence.sample_rate).toFixed(2)} s`;
    const restore = document.createElement("button");
    restore.type = "button";
    restore.textContent = "Restore proposal";
    restore.addEventListener("click", async () => {
      const capture = state.capture;
      if (!capture || state.liveView) return;
      restore.disabled = true;
      restore.textContent = "Restoring…";
      try {
        const restored = await saveCvProposalDecision(capture.id, detectionFromCvDecision(decision), "restored");
        if (state.capture !== capture) return;
        state.cvRejectedProposals = state.cvRejectedProposals.filter((item) => item.proposal_id !== restored.proposal_id);
        const detection = detectionFromCvDecision(restored);
        state.cvDetections = [...state.cvDetections, detection].sort((left, right) => left.sample_start - right.sample_start);
        renderCvDetections();
        renderCvDismissedProposals();
        drawOverlays();
        el("cv-status").textContent = "Dismissal restored; proposal is visible in this scan again. It remains provisional and is not a training label.";
      } catch (error) {
        if (state.capture !== capture) return;
        restore.disabled = false;
        restore.textContent = "Restore proposal";
        el("cv-status").textContent = `Could not restore proposal · ${error.message}`;
      }
    });
    row.append(summary, detail, restore);
    list.append(row);
  }
}

function matchingSavedRegionForCandidate(candidate) {
  const proposalId = candidate?.cv_proposal_id || candidate?.proposal_id;
  const active = state.regions.filter((region) => region.lineage_state !== "superseded"
    && !state.windowRegionDeletes.has(region.id)
    && !pendingWindowMergeSources().regions.has(region.id));
  if (proposalId) {
    const byProposal = active.find((region) =>
      region.evidence?.detector_proposal?.proposal_id === proposalId);
    if (byProposal) return byProposal;
  }
  const bounds = candidate?._originalProposalBounds || candidate;
  if (!Number.isFinite(bounds?.sample_start) || !Number.isFinite(bounds?.sample_count)
    || !Number.isFinite(bounds?.frequency_low_hz) || !Number.isFinite(bounds?.frequency_high_hz)) return null;
  const exact = active.filter((region) => {
    const geometry = state.windowRegionEdits.get(region.id)?.geometry || region;
    return geometry.sample_start === bounds.sample_start
      && geometry.sample_count === bounds.sample_count
      && Math.abs(geometry.frequency_low_hz - bounds.frequency_low_hz) < 0.01
      && Math.abs(geometry.frequency_high_hz - bounds.frequency_high_hz) < 0.01;
  });
  return exact.length === 1 ? exact[0] : null;
}

function selectSavedRegionFromCandidate(region, candidate) {
  selectSavedRegion(region);
  const original = candidate?._originalProposalBounds;
  if (!original) return;
  const bounds = candidate;
  const wasRedrawn = ["sample_start", "sample_count", "frequency_low_hz", "frequency_high_hz"]
    .some((field) => bounds[field] !== original[field]);
  if (!wasRedrawn) return;
  state.selection = {
    ...state.selection,
    sample_start: bounds.sample_start,
    sample_count: bounds.sample_count,
    frequency_low_hz: bounds.frequency_low_hz,
    frequency_high_hz: bounds.frequency_high_hz,
  };
  reprojectSelection();
  saveSelectedExistingRegionDraft();
  state.selectionMergeRegions = boxesCrossedBySelection(state.selection, region.id);
  state.geometryMergeRegions = [...state.selectionMergeRegions];
  updateSelectionMergeControls();
  drawOverlays();
}

function selectCvDetection(detection) {
  const data = state.waterfall;
  if (!data || state.liveView) return;
  const savedRegion = matchingSavedRegionForCandidate(detection);
  if (savedRegion) {
    selectSavedRegionFromCandidate(savedRegion, detection);
    el("inspector-context").textContent = `Saved CV region · ${savedRegion.signal_id || "signal identity"} · revise its existing label or bounds.`;
    return;
  }
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
    detector_evidence: detection.detector_evidence || detectorEvidenceFor(detection),
  };
  selectTrack(proposal);
  el("cv-presence-row").hidden = false;
  el("cv-presence-confirmed").checked = false;
  el("inspector-context").textContent = `Provisional CV box · ${(detection.confidence * 100).toFixed(1)}% detector score · signal identity unknown`;
  el("selection-details").textContent += " · inspect bounds, then run the existing classifier or save as unknown";
}

function updateSingleBoxButton() {
  const button = el("draw-signal-box");
  if (button) {
    button.textContent = state.drawingSignalBox ? "Cancel box drawing" : "Draw signal box";
    button.setAttribute("aria-pressed", String(state.drawingSignalBox));
  }
  const batchButton = el("batch-box-toggle");
  if (batchButton) {
    batchButton.setAttribute("aria-pressed", String(state.batchBoxMode));
    batchButton.textContent = state.batchBoxMode ? "Done drawing" : state.batchBoxes.length ? "Add boxes" : "Draw boxes";
  }
}

function cancelSingleBoxDrawing(message = "Box drawing cancelled.") {
  state.drawingSignalBox = false;
  updateSingleBoxButton();
  el("box-draw-status").textContent = message;
  drawOverlays();
}

function armCandidateRedraw(candidate) {
  if (!state.waterfall || state.liveView || state.coverageOnly) return;
  saveSelectedBatchLabelDraft();
  state.drawingSignalBox = false;
  updateSingleBoxButton();
  if (state.cvDetections.includes(candidate) && !candidate._originalProposalBounds) {
    candidate._originalProposalBounds = {
      sample_start: candidate.sample_start,
      sample_count: candidate.sample_count,
      frequency_low_hz: candidate.frequency_low_hz,
      frequency_high_hz: candidate.frequency_high_hz,
    };
  }
  state.adjustingCandidate = candidate;
  el("cv-status").textContent = "Redrawing candidate bounds · drag the corrected time/frequency rectangle on the waterfall. It can extend beyond the detector box; the original detection evidence is retained.";
  el("hint").classList.add("hidden");
}

el("draw-signal-box").addEventListener("click", () => {
  if (!state.waterfall || state.liveView || state.coverageOnly) return;
  if (state.drawingSignalBox) {
    cancelSingleBoxDrawing();
    return;
  }
  if (state.batchBoxMode) {
    state.batchBoxMode = false;
    renderBatchBoxes();
  }
  saveSelectedBatchLabelDraft();
  state.selection = null;
  state.classifierProposal = null;
  beginNewRegion();
  state.adjustingCandidate = null;
  state.drawingSignalBox = true;
  updateSingleBoxButton();
  el("box-draw-status").textContent = "Drawing mode active · drag a rectangle on the waterfall. Save the selection as a region in the label panel.";
  updateSceneReviewAvailability();
  drawOverlays();
  el("hint").classList.add("hidden");
});

function renderBatchBoxes() {
  const list = el("batch-box-list");
  list.replaceChildren();
  el("batch-box-summary").textContent = `${state.batchBoxes.length} ${state.batchBoxes.length === 1 ? "box" : "boxes"}`;
  state.batchBoxes.forEach((box, index) => {
    const row = document.createElement("div");
    row.className = `batch-box-item${state.batchReviewBox === box ? " is-selected" : ""}`;
    const title = document.createElement("strong");
    const savedLabel = box.labelDraft?.mode_name
      || [box.labelDraft?.variant, box.labelDraft?.protocol, box.labelDraft?.method, box.labelDraft?.family].find(Boolean);
    const identityState = box.labelDraft?.identity_unknown_confirmed ? "Unknown" : savedLabel || "needs label";
    title.textContent = `Signal ${index + 1} · ${identityState}`;
    const time = document.createElement("span");
    const start = box.sample_start / state.capture.sample_rate;
    const end = (box.sample_start + box.sample_count) / state.capture.sample_rate;
    time.textContent = `${start.toFixed(2)}–${end.toFixed(2)} s · ${box.frequency_low_hz.toFixed(0)}–${box.frequency_high_hz.toFixed(0)} Hz`;
    row.append(title, time);
    if (box.result) {
      const suggestion = document.createElement("span");
      suggestion.textContent = box.error
        ? `Classification failed · ${box.error}`
        : box.result.choice === "unknown" || (!box.result.family && !box.result.method)
          ? `No confident suggestion${box.result.top_candidate ? ` · top guess ${box.result.top_candidate}` : ""}`
          : `Suggested ${box.result.method || box.result.family}${box.result.family && box.result.method ? ` · ${box.result.family}` : ""}${Number.isFinite(box.result.confidence) ? ` · ${(100 * box.result.confidence).toFixed(1)}%` : ""}`;
      row.append(suggestion);
    }
    const review = document.createElement("button");
    review.type = "button";
    review.textContent = state.batchReviewBox === box ? "Selected for labeling" : "Select to label";
    review.setAttribute("aria-pressed", String(state.batchReviewBox === box));
    review.disabled = state.batchClassifying;
    review.addEventListener("click", () => reviewBatchBox(index));
    row.append(review);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove box";
    remove.disabled = state.batchClassifying;
    remove.addEventListener("click", () => {
      const wasSelected = state.batchReviewBox === box;
      if (wasSelected) clearCurrentSelection("Draft signal removed.");
      if (box.draftMergeId) {
        const undoIndex = state.batchBoxUndo.findLastIndex((change) => change.type === "undo-merge" && change.box === box);
        const stagedUndo = undoIndex >= 0 ? state.batchBoxUndo.splice(undoIndex, 1)[0] : null;
        state.batchBoxUndo = state.batchBoxUndo.filter((change) => change.box !== box);
        if (stagedUndo) {
          state.batchBoxes = stagedUndo.previousBoxes;
          state.windowMerges = stagedUndo.previousMerges;
          for (const edit of stagedUndo.priorSourceEdits || []) state.windowRegionEdits.set(edit.region_id, edit);
          state.batchReviewBox = stagedUndo.previousSelection || null;
          state.editRegion = stagedUndo.previousEditRegion || null;
          state.selection = stagedUndo.previousGeometrySelection ? {...stagedUndo.previousGeometrySelection} : null;
        } else state.batchBoxes.splice(index, 1);
      } else {
        state.batchBoxUndo.push({type: "remove", box, index});
        state.batchBoxes.splice(index, 1);
      }
      renderBatchBoxes();
      renderRegions();
      state.selectionMergeRegions = state.selection ? boxesCrossedBySelection(state.selection, state.editRegion?.id) : [];
      updateSelectionMergeControls();
      drawOverlays();
    });
    row.append(remove);
    list.append(row);
  });
  el("batch-box-classify").disabled = state.batchClassifying || !state.batchBoxes.length || !state.modelAvailable || state.batchBoxMode || state.batchBoxes.some((box) => box.sample_count < 1024);
  el("batch-box-clear").disabled = state.batchClassifying || !state.batchBoxes.length;
  el("batch-box-toggle").disabled = state.batchClassifying || !state.capture || state.liveView || state.coverageOnly;
  el("batch-box-toggle").textContent = state.batchBoxMode ? "Done drawing" : state.batchBoxes.length ? "Add boxes" : "Draw boxes";
  updateSingleBoxButton();
  el("batch-box-status").textContent = state.batchClassifying
    ? `Getting independent classifier suggestions for ${state.batchBoxes.length} draft signals…`
    : state.batchBoxMode
      ? `Drawing mode active · drag over each signal. ${state.batchBoxes.length} box${state.batchBoxes.length === 1 ? "" : "es"} marked so far.`
      : state.batchBoxes.some((box) => box.sample_count < 1024)
        ? `${state.batchBoxes.length} box${state.batchBoxes.length === 1 ? "" : "es"} marked · each box needs at least 1,024 samples.`
        : !state.modelAvailable
          ? `${state.batchBoxes.length} box${state.batchBoxes.length === 1 ? "" : "es"} marked · no classifier suggestion is loaded, but you can assign labels manually.`
        : state.batchBoxes.length || state.windowRegionEdits.size || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length
          ? `${state.batchBoxes.length} new · ${state.windowRegionEdits.size} edited · ${state.windowRegionDeletes.size} to remove · ${state.windowMerges.length} merge${state.windowMerges.length === 1 ? "" : "s"} · ${state.windowIdentityLinks.length} identity link${state.windowIdentityLinks.length === 1 ? "" : "s"} staged · save the window changes together.`
          : "Draw a box around each signal, then label and save the window together.";
  updateAnnotationSaveAvailability();
}

function readBatchLabelDraft() {
  return {
    mode_id: state.labelModeId || null,
    mode_name: el("mode-search").value.trim(),
    family: el("family").value.trim(),
    method: el("method").value.trim(),
    protocol: el("protocol").value.trim(),
    variant: el("variant").value.trim(),
    notes: el("notes").value.trim(),
    identity_unknown_confirmed: el("batch-unknown-confirmed").checked,
  };
}

function saveSelectedBatchLabelDraft() {
  if (state.batchReviewBox) {
    const box = state.batchReviewBox;
    const previous = box.labelDraft || {};
    const next = readBatchLabelDraft();
    if (JSON.stringify(previous) !== JSON.stringify(next)) {
      const lastChange = state.batchBoxUndo.at(-1);
      if (lastChange?.type !== "restore-box-label" || lastChange.box !== box) {
        state.batchBoxUndo.push({type: "restore-box-label", box, labelDraft: {...previous}});
      }
      box.labelDraft = next;
    }
  } else saveSelectedExistingRegionDraft();
}

function stageCurrentSelectionAsWindowDraft() {
  if (!state.selection || state.editRegion || state.batchReviewBox || state.batchBoxMode || !state.capture) return null;
  const selection = {...state.selection};
  const selectedTrack = state.selectedTrack;
  const detectorEvidence = selectedTrack?.detector_evidence
    || (selectedTrack?.cv_proposal_id ? detectorEvidenceFor(selectedTrack) : null);
  const proposal = state.classifierProposal?.capture_id === state.capture.id
    && state.classifierProposal.selection === state.selection
    ? state.classifierProposal.result : null;
  const box = {
    client_id: crypto.randomUUID(),
    ...selection,
    labelDraft: readBatchLabelDraft(),
    origin: detectorEvidence ? "detector_proposal" : el("missed-signal").checked ? "user_added_missed_detection" : "manual",
    detector_evidence: detectorEvidence,
    signal_presence_confirmed: Boolean(detectorEvidence && el("cv-presence-confirmed").checked),
    classifier_proposal: proposal ? {
      sample_start: proposal.sample_start,
      sample_count: Math.min(1_000_000, proposal.sample_count),
      selection_sample_count: selection.sample_count,
      frequency_low_hz: proposal.selection_frequency_range_hz?.[0] ?? selection.frequency_low_hz,
      frequency_high_hz: proposal.selection_frequency_range_hz?.[1] ?? selection.frequency_high_hz,
      crop_count: proposal.requested_crop_count || 1,
    } : null,
  };
  state.batchBoxes.push(box);
  state.batchBoxUndo.push({type: "add", box});
  state.batchReviewBox = box;
  state.selectedTrack = null;
  state.classifierProposal = null;
  state.selectionMergeRegions = boxesCrossedBySelection(state.selection);
  updateSelectionMergeControls();
  restoreBatchLabelDraft(box);
  renderBatchBoxes();
  drawOverlays();
  return box;
}

function saveSelectedExistingRegionDraft() {
  const edit = state.editRegion;
  if (!edit || !state.selection || state.batchReviewBox) return;
  const region = state.regions.find((item) => item.id === edit.id);
  if (!region) return;
  const geometry = {
    sample_start: state.selection.sample_start,
    sample_count: state.selection.sample_count,
    frequency_low_hz: state.selection.frequency_low_hz,
    frequency_high_hz: state.selection.frequency_high_hz,
  };
  const base = state.windowRegionEdits.get(region.id) || {
    region_id: region.id,
    expected_revision: edit.revision,
    expected_geometry_revision: edit.geometry_revision,
    geometry: {
      sample_start: region.sample_start,
      sample_count: region.sample_count,
      frequency_low_hz: region.frequency_low_hz,
      frequency_high_hz: region.frequency_high_hz,
    },
    labelDraft: {
      mode_id: region.mode_id || null,
      mode_name: region.mode_id || "",
      family: region.family || "",
      method: region.method || "",
      protocol: region.protocol || "",
      variant: region.variant || "",
      notes: region.notes || "",
      identity_unknown_confirmed: Boolean(region.evidence?.identity_unknown_confirmed),
    },
  };
  const previousDraft = state.windowRegionEdits.has(region.id)
    ? JSON.parse(JSON.stringify(state.windowRegionEdits.get(region.id)))
    : null;
  const previousFingerprint = previousDraft ? JSON.stringify(previousDraft) : "";
  base.geometry = geometry;
  base.labelDraft = readBatchLabelDraft();
  const sameGeometry = geometry.sample_start === region.sample_start
    && geometry.sample_count === region.sample_count
    && geometry.frequency_low_hz === region.frequency_low_hz
    && geometry.frequency_high_hz === region.frequency_high_hz;
  const sameLabel = base.labelDraft.mode_id === (region.mode_id || null)
    && ["family", "method", "protocol", "variant", "notes"].every((field) => base.labelDraft[field] === (region[field] || ""))
    && base.labelDraft.identity_unknown_confirmed === Boolean(region.evidence?.identity_unknown_confirmed);
  if (sameGeometry && sameLabel) state.windowRegionEdits.delete(region.id);
  else state.windowRegionEdits.set(region.id, base);
  const nextDraft = state.windowRegionEdits.get(region.id) || null;
  const nextFingerprint = nextDraft ? JSON.stringify(nextDraft) : "";
  if (previousFingerprint !== nextFingerprint) {
    state.batchBoxUndo.push({type: "restore-region-edit", regionId: region.id, previousDraft});
  }
  updateAnnotationSaveAvailability();
  renderRegions();
}

function windowRegionUpdatePayloads() {
  return [...state.windowRegionEdits.values()].map((edit) => ({
    region_id: edit.region_id,
    expected_revision: edit.expected_revision,
    expected_geometry_revision: edit.expected_geometry_revision,
    ...edit.geometry,
    mode_id: edit.labelDraft.mode_id || null,
    family: edit.labelDraft.family || null,
    method: edit.labelDraft.method || null,
    protocol: edit.labelDraft.protocol || null,
    variant: edit.labelDraft.variant || null,
    notes: edit.labelDraft.notes || "",
    identity_unknown_confirmed: Boolean(edit.labelDraft.identity_unknown_confirmed),
  }));
}

function windowRegionDeletionPayloads() {
  return [...state.windowRegionDeletes.values()].map((region) => ({
    region_id: region.id,
    expected_revision: region.revision,
    expected_geometry_revision: region.geometry_revision,
  }));
}

function windowMergePayloads() {
  return state.windowMerges.map((merge) => {
    const box = state.batchBoxes.find((item) => item.mergeClientId === merge.client_id);
    const draft = box?.labelDraft || merge.labelDraft || {};
    return {
      client_id: merge.client_id,
      source_region_ids: merge.source_region_ids,
      expected_revisions: merge.expected_revisions,
      expected_geometry_revisions: merge.expected_geometry_revisions,
      cv_proposals: merge.cv_proposals,
      heuristic_proposals: merge.heuristic_proposals || [],
      target_region_id: merge.target_region_id,
      sample_start: box?.sample_start ?? merge.sample_start,
      sample_count: box?.sample_count ?? merge.sample_count,
      frequency_low_hz: box?.frequency_low_hz ?? merge.frequency_low_hz,
      frequency_high_hz: box?.frequency_high_hz ?? merge.frequency_high_hz,
      mode_id: draft.mode_id || null,
      family: draft.family || null,
      method: draft.method || null,
      protocol: draft.protocol || null,
      variant: draft.variant || null,
      notes: draft.notes || "",
      identity_unknown_confirmed: Boolean(draft.identity_unknown_confirmed),
    };
  });
}

function windowIdentityLinkPayloads() {
  return state.windowIdentityLinks.map((link) => ({
    ...(link.signal_id ? {signal_id: link.signal_id} : {target_region_id: link.target_region_id}),
    region_ids: [...link.region_ids],
    expected_signal_ids: {...link.expected_signal_ids},
    expected_revisions: {...link.expected_revisions},
  }));
}

function refreshIdentityLinkControls() {
  const window = state.capture ? annotationWindowBounds(state.capture) : null;
  const selected = [...state.identityLinkSelection].filter((id) => {
    const region = state.regions.find((item) => item.id === id);
    return region && region.lineage_state !== "superseded" && !state.windowRegionDeletes.has(id)
      && window && region.sample_start >= window.startSample
      && region.sample_start + region.sample_count <= window.startSample + window.sampleCount;
  });
  state.identityLinkSelection = new Set(selected);
  const button = el("link-signal-identity");
  const undoButton = el("undo-signal-identity-link");
  const status = el("identity-link-status");
  if (!button || !status) return;
  button.disabled = selected.length < 2 || state.liveView || state.batchSaving;
  const currentButton = el("same-signal-current");
  if (currentButton) currentButton.disabled = !state.editRegion || state.liveView || state.batchSaving;
  if (undoButton) undoButton.disabled = !state.windowIdentityLinks.length || state.batchSaving;
  status.textContent = selected.length
    ? `${selected.length} boxes selected · they will remain separate regions.${state.windowIdentityLinks.length ? ` ${state.windowIdentityLinks.length} identity link${state.windowIdentityLinks.length === 1 ? "" : "s"} staged.` : ""}`
    : state.windowIdentityLinks.length
      ? `${state.windowIdentityLinks.length} identity link${state.windowIdentityLinks.length === 1 ? "" : "s"} staged · save annotations to apply.`
      : "Select two or more saved signals below.";
  for (const row of document.querySelectorAll(".region-row[data-region-id]")) {
    row.classList.toggle("identity-link-selected", state.identityLinkSelection.has(row.dataset.regionId));
  }
}

el("link-signal-identity")?.addEventListener("click", () => {
  const selected = [...state.identityLinkSelection].map((id) => state.regions.find((region) => region.id === id)).filter(Boolean);
  if (selected.length < 2) return;
  openSameSignalDialog(selected.map((region) => region.id));
});

function openSameSignalDialog(regionIds) {
  const ids = [...new Set(regionIds)];
  const tbody = el("same-signal-parent-rows");
  tbody.replaceChildren();
  const choices = state.regions.filter((region) => region.lineage_state !== "superseded" && !state.windowRegionDeletes.has(region.id));
  const rate = state.capture?.sample_rate || 1;
  for (const region of choices) {
    const row = document.createElement("tr");
    const radioCell = document.createElement("td");
    const radio = document.createElement("input");
    radio.type = "radio"; radio.name = "same-signal-parent"; radio.value = region.id;
    radio.disabled = ids.length > 1 ? !ids.includes(region.id) : region.id === ids[0];
    radio.checked = ids.length > 1 && ids.includes(region.id) && region.id === ids[0];
    radio.addEventListener("change", () => { el("same-signal-confirm").disabled = false; });
    radioCell.append(radio); row.append(radioCell);
    const label = [region.family, region.method, region.protocol, region.variant].filter(Boolean).join(" · ") || region.mode_name || "Needs label";
    const source = [region.source, region.evidence?.detector_proposal ? "CV" : "", region.evidence?.classifier_proposal ? "Classifier" : ""].filter(Boolean).join(" · ");
    const band = Number.isFinite(region.frequency_low_hz) && Number.isFinite(region.frequency_high_hz) ? `${region.frequency_low_hz.toFixed(0)}–${region.frequency_high_hz.toFixed(0)} Hz` : "Unknown";
    for (const value of [region.signal_id || "New identity", label, source || "Saved", `${(region.sample_start / rate).toFixed(2)}–${((region.sample_start + region.sample_count) / rate).toFixed(2)} s`, band]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    row.addEventListener("click", (event) => { if (event.target !== radio) { radio.checked = true; radio.dispatchEvent(new Event("change")); } });
    tbody.append(row);
  }
  window.identityModalRegionIds = ids;
  el("same-signal-dialog-summary").textContent = `${ids.length} box${ids.length === 1 ? "" : "es"} will keep separate bounds and share the parent’s identity and label.`;
  el("same-signal-confirm").disabled = !tbody.querySelector('input[name="same-signal-parent"]:checked');
  el("same-signal-dialog").showModal();
}

el("same-signal-current")?.addEventListener("click", () => {
  if (state.editRegion) openSameSignalDialog([state.editRegion.id]);
});
el("same-signal-cancel")?.addEventListener("click", () => el("same-signal-dialog").close());
el("same-signal-confirm")?.addEventListener("click", () => {
  const target = state.regions.find((region) => region.id === document.querySelector('input[name="same-signal-parent"]:checked')?.value);
  const selected = (window.identityModalRegionIds || []).map((id) => state.regions.find((region) => region.id === id)).filter(Boolean);
  if (!target || !selected.length) return;
  if (!selected.some((region) => region.id === target.id)) selected.push(target);
  const signalIds = new Set(state.windowIdentityLinks.flatMap((link) => link.region_ids));
  if (selected.some((region) => signalIds.has(region.id))) {
    el("identity-link-status").textContent = "A selected box is already part of another pending identity link. Save or clear that link first.";
    return;
  }
  state.windowIdentityLinks.push({
    target_region_id: target.id,
    region_ids: selected.map((region) => region.id),
    expected_signal_ids: Object.fromEntries(selected.map((region) => [region.id, region.signal_id])),
    expected_revisions: Object.fromEntries(selected.map((region) => [region.id, region.revision])),
  });
  el("identity-link-status").textContent = `Link staged to ${target.signal_id || "the selected parent"}. Save annotations to apply it.`;
  state.identityLinkSelection.clear();
  renderRegions();
  updateAnnotationSaveAvailability();
  el("same-signal-dialog").close();
});

el("undo-signal-identity-link")?.addEventListener("click", () => {
  if (!state.windowIdentityLinks.length) return;
  state.windowIdentityLinks.pop();
  el("identity-link-status").textContent = "Last identity link removed from the window draft.";
  refreshIdentityLinkControls();
  updateAnnotationSaveAvailability();
});

document.getElementById("identity-signal-id")?.addEventListener("input", () => refreshIdentityLinkControls());

function pendingWindowMergeSources() {
  return {
    regions: new Set(state.windowMerges.flatMap((merge) => merge.source_region_ids)),
    proposals: new Set(state.windowMerges.flatMap((merge) => merge.cv_proposals.map((item) => item.proposal_id))),
    tracks: new Set(state.windowMerges.flatMap((merge) => (merge.heuristic_proposals || []).map((item) => item.candidate_id))),
  };
}

function restoreBatchLabelDraft(box) {
  const draft = box.labelDraft || {};
  state.labelModeId = draft.mode_id || null;
  state.applyingMode = true;
  el("mode-search").value = draft.mode_name || "";
  for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = draft[field] || "";
  el("batch-unknown-confirmed").checked = Boolean(draft.identity_unknown_confirmed);
  el("batch-unknown-row").hidden = false;
  el("mode-resolution").textContent = draft.mode_id
    ? "Saved label selection for this draft."
    : draft.identity_unknown_confirmed
      ? "This is a confirmed signal with unknown identity."
      : "Choose a known mode, enter a partial label, or leave it unlabeled.";
  state.applyingMode = false;
}

function batchBoxLabelText(box, index) {
  const draft = box.labelDraft;
  const label = draft?.identity_unknown_confirmed
    ? "Unknown"
    : draft?.mode_name || [draft?.variant, draft?.protocol, draft?.method, draft?.family].find(Boolean);
  return label ? `Draft ${index + 1} · ${label}` : `Draft ${index + 1}`;
}

function regionIdentityCaption(region, label = region, labelDraft = null) {
  const identity = [label?.variant, label?.protocol, label?.method, label?.family, label?.mode_name, label?.mode_id]
    .find(Boolean);
  if (identity) return identity;
  const explicitlyUnknown = labelDraft?.identity_unknown_confirmed
    ?? label?.identity_unknown_confirmed
    ?? Boolean(region?.evidence?.identity_unknown_confirmed);
  return explicitlyUnknown ? "Unknown" : "Needs label";
}

function updateAnnotationSaveAvailability() {
  const annotate = state.inspectorMode === "annotate";
  const hasWindowChanges = state.batchBoxes.length > 0 || state.windowRegionEdits.size > 0
    || state.windowRegionDeletes.size > 0 || state.windowMerges.length > 0 || state.windowIdentityLinks.length > 0;
  const hasWindowDrafts = hasWindowChanges && !state.liveView && !state.batchSaving;
  el("save-annotations").disabled = hasWindowDrafts
    ? false
    : !state.selection || !state.capture || state.liveView || state.batchSaving;
  el("save-annotations").textContent = state.liveView
    ? "Save annotations"
    : hasWindowChanges ? "Save all annotations" : annotate ? "Save annotations" : "Save signal";
  el("save-annotations").title = state.liveView
    ? state.selection
      ? "Save this labeled signal as a training candidate from the current 10-second live window."
      : "Select a signal fully inside a complete 10-second live window to save a training candidate."
    : hasWindowChanges
      ? "Save new boxes, labels, geometry changes, identity links, merges, and removals together for this 10-second window."
      : annotate
        ? "Save this annotation as part of the current 10-second window."
        : "Save the selected signal and its label.";
  el("save-annotations").hidden = annotate;
  el("save-annotations-close").disabled = !state.capture || state.liveView || state.batchSaving;
  const reportButton = el("report-window-pdf");
  if (reportButton) reportButton.disabled = !state.capture || !state.waterfall || state.liveView;
  el("save-annotations-complete").disabled = state.batchSaving || !state.capture || state.liveView;
}

function reportRegionsForWindow(bounds) {
  const inWindow = (item) => item.sample_start < bounds.endSample
    && item.sample_start + item.sample_count > bounds.startSample;
  const saved = state.regions.filter((region) => region.lineage_state !== "superseded"
    && !state.windowRegionDeletes.has(region.id) && inWindow(region)).map((region) => {
    const edit = state.windowRegionEdits.get(region.id);
    return {...region, ...(edit?.geometry || {}), ...(edit?.labelDraft || {}), reportDraft: Boolean(edit)};
  });
  const drafts = state.batchBoxes.filter(inWindow).map((box, index) => ({
    ...box, id: box.client_id || `draft-${index + 1}`, signal_id: null,
    reportDraft: true, source: "Manual draft", ...(box.labelDraft || {}),
  }));
  return [...saved, ...drafts].sort((a, b) => a.sample_start - b.sample_start || a.frequency_low_hz - b.frequency_low_hz);
}

function reportWaterfallCanvas(bounds, regions, cropRegion = null) {
  const data = state.waterfall;
  const rowIndices = data.sample_starts.map((sample, index) => ({sample, index}))
    .filter(({sample}) => sample < bounds.endSample && sample + data.fft_size > bounds.startSample);
  const canvas = document.createElement("canvas");
  if (!rowIndices.length) return canvas;
  const hasBand = cropRegion && Number.isFinite(cropRegion.frequency_low_hz) && Number.isFinite(cropRegion.frequency_high_hz);
  const frequencyStart = hasBand ? Math.max(0, lowerBound(data.frequency_hz, cropRegion.frequency_low_hz)) : 0;
  const frequencyEnd = hasBand ? Math.min(data.frequency_hz.length, lowerBound(data.frequency_hz, cropRegion.frequency_high_hz) + 1) : data.frequency_hz.length;
  const selectedRows = (cropRegion ? rowIndices.filter(({sample}) => sample < cropRegion.sample_start + cropRegion.sample_count && sample + data.fft_size > cropRegion.sample_start) : rowIndices).reverse();
  canvas.width = Math.max(1, frequencyEnd - frequencyStart);
  canvas.height = Math.max(1, selectedRows.length);
  const context = canvas.getContext("2d", {alpha:false});
  const image = context.createImageData(canvas.width, canvas.height);
  const palette = state.palette || (state.palette = Array.from({length:256}, (_, i) => colorFor(-95 + i * 75 / 255, -95, -20)));
  selectedRows.forEach(({index}, y) => {
    for (let x = 0; x < canvas.width; x++) {
      const level = Math.max(0, Math.min(255, Math.round((data.db[index][frequencyStart + x] + 95) * 255 / 75)));
      const [r, g, b] = palette[level];
      const offset = (y * canvas.width + x) * 4;
      image.data[offset] = r; image.data[offset + 1] = g; image.data[offset + 2] = b; image.data[offset + 3] = 255;
    }
  });
  context.putImageData(image, 0, 0);
  if (!cropRegion) {
    context.strokeStyle = "#ffffff"; context.lineWidth = Math.max(2, Math.round(canvas.width / 900));
    context.font = `${Math.max(11, Math.round(canvas.width / 110))}px sans-serif`;
    for (const region of regions) {
      if (!Number.isFinite(region.frequency_low_hz) || !Number.isFinite(region.frequency_high_hz)) continue;
      const x1 = Math.max(0, (region.frequency_low_hz - data.frequency_hz[frequencyStart]) / Math.max(1, data.frequency_hz[frequencyEnd - 1] - data.frequency_hz[frequencyStart]) * canvas.width);
      const x2 = Math.min(canvas.width - 1, (region.frequency_high_hz - data.frequency_hz[frequencyStart]) / Math.max(1, data.frequency_hz[frequencyEnd - 1] - data.frequency_hz[frequencyStart]) * canvas.width);
      const y1 = Math.max(0, Math.min(canvas.height - 1, (bounds.endSample - Math.min(bounds.endSample, region.sample_start + region.sample_count)) / Math.max(1, bounds.sampleCount) * canvas.height));
      const y2 = Math.max(0, Math.min(canvas.height - 1, (bounds.endSample - Math.max(bounds.startSample, region.sample_start)) / Math.max(1, bounds.sampleCount) * canvas.height));
      context.strokeStyle = "#f6d365"; context.strokeRect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
      const caption = [region.signal_id, region.method || region.mode_name || region.family].filter(Boolean).join(" · ") || "Unlabeled";
      context.fillStyle = "#08111de8"; context.fillRect(x1, Math.max(0, y1 - 16), Math.min(canvas.width - x1, context.measureText(caption).width + 8), 16);
      context.fillStyle = "#ffffff"; context.fillText(caption, x1 + 4, Math.max(12, y1 - 4));
    }
  }
  return canvas;
}

function reportSnrEstimate(region) {
  const data = state.waterfall;
  if (!Number.isFinite(region.frequency_low_hz) || !Number.isFinite(region.frequency_high_hz)) return "Unavailable (frequency bounds unknown)";
  const f0 = Math.max(0, lowerBound(data.frequency_hz, region.frequency_low_hz));
  const f1 = Math.min(data.frequency_hz.length, lowerBound(data.frequency_hz, region.frequency_high_hz) + 1);
  const signal = [], noise = [];
  for (let i = 0; i < data.sample_starts.length; i++) {
    const sample = data.sample_starts[i];
    if (sample >= region.sample_start + region.sample_count || sample + data.fft_size <= region.sample_start) continue;
    for (let f = f0; f < f1; f++) signal.push(data.db[i][f]);
    const width = Math.max(1, f1 - f0);
    for (let f = Math.max(0, f0 - width); f < f0; f++) noise.push(data.db[i][f]);
    for (let f = f1; f < Math.min(data.frequency_hz.length, f1 + width); f++) noise.push(data.db[i][f]);
  }
  const median = (values) => { if (!values.length) return null; values.sort((a, b) => a - b); return values[Math.floor(values.length / 2)]; };
  const signalLevel = median(signal), noiseLevel = median(noise);
  if (signalLevel === null || noiseLevel === null) return "Unavailable";
  return `${(signalLevel - noiseLevel).toFixed(1)} dB estimated band contrast (local waterfall median)`;
}

function escapeReportText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}

function generateWindowPdfReport() {
  if (!state.capture || !state.waterfall || state.liveView) return;
  const popup = window.open("", "_blank");
  if (!popup) { showError("Allow popups to open the printable window report."); return; }
  const bounds = annotationWindowBounds();
  const regions = reportRegionsForWindow(bounds);
  const fullImage = reportWaterfallCanvas(bounds, regions).toDataURL("image/png");
  const rate = state.capture.sample_rate;
  const start = (bounds.startSample / rate).toFixed(2), end = (bounds.endSample / rate).toFixed(2);
  const details = regions.map((region, index) => {
    const crop = reportWaterfallCanvas(bounds, regions, region).toDataURL("image/png");
    const mode = region.mode_name || region.method || region.variant || region.protocol || region.family || "Unclassified";
    const time = `${(region.sample_start / rate).toFixed(2)}–${((region.sample_start + region.sample_count) / rate).toFixed(2)} s`;
    const frequency = Number.isFinite(region.frequency_low_hz) && Number.isFinite(region.frequency_high_hz)
      ? `${region.frequency_low_hz.toFixed(0)}–${region.frequency_high_hz.toFixed(0)} Hz` : "Unknown";
    const confidence = region.confidence ?? region.evidence?.classifier_proposal?.confidence;
    const confidenceLine = Number.isFinite(confidence) ? `<p><b>Classifier confidence:</b> ${(confidence * 100).toFixed(1)}%</p>` : "";
    const taxonomy = [["Family", region.family], ["Method", region.method], ["Protocol", region.protocol], ["Variant", region.variant]]
      .filter(([, value]) => value).map(([name, value]) => `<span><b>${name}:</b> ${escapeReportText(value)}</span>`).join(" ");
    return `<article class="region"><h2>Region ${index + 1}${region.signal_id ? ` · ${escapeReportText(region.signal_id)}` : ""}${region.reportDraft ? " · Draft" : ""}</h2><img class="closeup" src="${crop}" alt="Waterfall closeup for region ${index + 1}"><div class="facts"><p><b>Classification:</b> ${escapeReportText(mode)}</p><p>${taxonomy || "No detailed taxonomy saved"}</p>${confidenceLine}<p><b>Time:</b> ${time} · <b>Frequency:</b> ${frequency}</p><p><b>Source:</b> ${escapeReportText(region.source || "Manual annotation")} · <b>Review:</b> ${escapeReportText(region.review_state || "Draft")}</p><p><b>SNR:</b> ${escapeReportText(reportSnrEstimate(region))}</p>${region.notes ? `<p><b>Notes:</b> ${escapeReportText(region.notes)}</p>` : ""}</div></article>`;
  }).join("");
  popup.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>Signal window report · ${escapeReportText(state.capture.display_name || state.capture.id)}</title><style>@page{size:letter;margin:14mm}body{font:12px Arial,sans-serif;color:#17212b;margin:0}h1{font-size:22px;margin:0 0 8px}h2{font-size:16px;margin:0 0 8px}.meta{color:#485967;margin:4px 0 14px}.window{width:100%;max-height:220mm;object-fit:contain}.region{page-break-before:always;break-before:page}.closeup{display:block;width:100%;max-height:125mm;object-fit:contain;background:#08111b}.facts{display:grid;grid-template-columns:1fr 1fr;gap:2px 16px}.facts p{margin:5px 0}.foot{font-size:10px;color:#52616c;margin-top:12px}@media screen{body{max-width:850px;margin:24px auto;padding:24px}.window,.closeup{border:1px solid #ccd5dc}.print{position:fixed;top:12px;right:12px;padding:10px 16px}}@media print{.print{display:none}}</style></head><body><button class="print" onclick="window.print()">Print / Save as PDF</button><h1>Signal window report</h1><p class="meta"><b>${escapeReportText(state.capture.display_name || state.capture.id)}</b> · Window ${bounds.index + 1} of ${bounds.total} · ${start}–${end} s · ${regions.length} region${regions.length === 1 ? "" : "s"}</p><img class="window" src="${fullImage}" alt="Annotated full 10-second waterfall window"><p class="foot">SNR is estimated as median in-band waterfall level minus median adjacent-bin level; it is a visual estimate, not a calibrated receiver measurement. Draft markup is identified above.</p>${details || "<article class=region><p>No saved or draft regions are in this window.</p></article>"}</body></html>`);
  popup.document.close();
  popup.focus();
  popup.setTimeout(() => popup.print(), 350);
}

el("report-window-pdf")?.addEventListener("click", generateWindowPdfReport);

async function saveWindowAnnotationDrafts() {
  const capture = state.capture;
  if (!capture || state.liveView) return null;
  saveSelectedBatchLabelDraft();
  if (!state.batchBoxes.length && !state.windowRegionEdits.size && !state.windowRegionDeletes.size && !state.windowMerges.length && !state.windowIdentityLinks.length) return null;
  const annotationWindow = annotationWindowBounds(capture);
  const windowStart = annotationWindow.startSample;
  const windowCount = annotationWindow.sampleCount;
  if (state.batchBoxes.some((box) => box.sample_start < windowStart || box.sample_start + box.sample_count > windowStart + windowCount)) {
    throw new Error("These boxes span beyond the current 10-second window. Save or review one 10-second window at a time.");
  }
  const annotations = state.batchBoxes.filter((box) => !box.mergeClientId).map((box) => {
    const draft = box.labelDraft || {};
    return {
      client_id: box.client_id,
      sample_start: box.sample_start,
      sample_count: box.sample_count,
      frequency_low_hz: box.frequency_low_hz,
      frequency_high_hz: box.frequency_high_hz,
      origin: box.origin || "manual",
      detector_evidence: box.detector_evidence || null,
      classifier_proposal: box.classifier_proposal || null,
      signal_presence_confirmed: Boolean(box.signal_presence_confirmed),
      mode_id: draft.mode_id || null,
      family: draft.family || null,
      method: draft.method || null,
      protocol: draft.protocol || null,
      variant: draft.variant || null,
      notes: draft.notes || "",
      identity_unknown_confirmed: Boolean(draft.identity_unknown_confirmed),
    };
  });
  const updates = windowRegionUpdatePayloads();
  const deletions = windowRegionDeletionPayloads();
  const merges = windowMergePayloads();
  const identityLinks = windowIdentityLinkPayloads();
  await ensureWindowSaveCapabilities({updates, deletions, merges, identityLinks});
  const contentDigest = JSON.stringify({windowStart, windowCount, annotations, updates, deletions, merges, identityLinks});
  if (!state.batchSaveKey || state.batchSaveDigest !== contentDigest) {
    state.batchSaveKey = crypto.randomUUID();
    state.batchSaveDigest = contentDigest;
  }
  state.batchSaving = true;
  updateAnnotationSaveAvailability();
  const changeCount = annotations.length + updates.length + deletions.length + merges.length + identityLinks.length;
  el("batch-box-status").textContent = `Saving ${changeCount} window change${changeCount === 1 ? "" : "s"} together…`;
  el("label-state").textContent = "Saving all annotations for this 10-second window…";
  try {
    const result = await request(`/api/captures/${encodeURIComponent(capture.id)}/regions/batch`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        idempotency_key: state.batchSaveKey,
        window_start_sample: windowStart,
        window_sample_count: windowCount,
        regions: annotations,
        updates,
        deletions,
        merges,
        identity_links: identityLinks,
      }),
    });
    if (state.capture?.id !== capture.id) return null;
    const mergeClientIds = new Set(merges.map((merge) => merge.client_id));
    const mergedTrackIds = new Set(merges.flatMap((merge) => (merge.heuristic_proposals || []).map((item) => item.candidate_id)));
    const mergedCvProposals = merges.flatMap((merge) => merge.cv_proposals || []);
    const mergedCvProposalIds = new Set(mergedCvProposals.map((proposal) => proposal.proposal_id).filter(Boolean));
    state.batchBoxes = [];
    state.batchBoxUndo = [];
    state.batchReviewBox = null;
    state.windowRegionEdits.clear();
    state.windowRegionDeletes.clear();
    state.windowMerges = [];
    state.windowIdentityLinks = [];
    state.identityLinkSelection.clear();
    state.batchSaveKey = null;
    state.batchSaveDigest = null;
    state.selection = null;
    state.tracks = state.tracks.filter((track) => !mergedTrackIds.has(heuristicCandidateId(track)));
    if (mergedCvProposalIds.size) {
      state.cvDetections = state.cvDetections.filter((detection) =>
        !mergedCvProposalIds.has(detection.cv_proposal_id || detection.proposal_id));
      const knownDismissedIds = new Set(state.cvRejectedProposals.map((proposal) => proposal.proposal_id));
      for (const proposal of mergedCvProposals) {
        if (!proposal.proposal_id || knownDismissedIds.has(proposal.proposal_id)) continue;
        const evidence = proposal.detector_evidence || detectorEvidenceFor(proposal);
        state.cvRejectedProposals.push({
          proposal_id: proposal.proposal_id,
          decision: "dismissed",
          sample_start: proposal.sample_start,
          sample_count: proposal.sample_count,
          frequency_low_hz: proposal.frequency_low_hz,
          frequency_high_hz: proposal.frequency_high_hz,
          confidence: proposal.confidence,
          model_sha256: evidence.model_sha256 || proposal.model_sha256,
          detector_evidence: evidence,
          note: "Merged into a saved signal; this is not a no-signal annotation.",
        });
        knownDismissedIds.add(proposal.proposal_id);
      }
      const cached = state.cvDetectionCache.get(capture.id);
      if (cached) state.cvDetectionCache.set(capture.id, {
        ...cached,
        detections: state.cvDetections,
        dismissed_proposals: state.cvRejectedProposals,
      });
      renderCvDetections();
      renderCvDismissedProposals();
      drawOverlays();
    }
    state.selectionMergeRegions = [];
    state.excludedMergeCandidates.clear();
    beginNewRegion();
    const savedCount = changeCount;
    const savedMessage = `${savedCount} window change${savedCount === 1 ? "" : "s"} saved together${result.replayed ? " (safe retry)" : ""}.`;
    el("batch-box-status").textContent = savedMessage;
    el("label-state").textContent = `${savedCount} signal change${savedCount === 1 ? "" : "s"} saved for this 10-second window.`;

    // The transaction is committed at this point. A later read failure must not
    // make the operator think the save failed or invite a duplicate operation.
    let refreshError = null;
    try {
      state.regions = await request(`/api/captures/${encodeURIComponent(capture.id)}/regions`);
      renderRegions();
      renderBatchBoxes();
      renderTracks();
      drawOverlays();
      const savedMerge = state.regions.find((region) => mergeClientIds.has(region.evidence?.window_annotation_merge?.client_id));
      if (savedMerge) selectSavedRegion(savedMerge, "label");
    } catch (error) {
      refreshError = error;
    }
    try {
      await refreshSceneCoverage(capture.id);
    } catch (error) {
      refreshError ||= error;
    }
    if (refreshError) {
      const message = `Annotations were saved, but the display could not refresh · ${refreshError.message}. Reopen this window to reload the saved signals.`;
      el("batch-box-status").textContent = message;
      el("label-state").textContent = message;
    }
    el("refresh-window-regions").hidden = true;
    return result;
  } catch (error) {
    if (error.status === 409) el("refresh-window-regions").hidden = false;
    throw error;
  } finally {
    state.batchSaving = false;
    updateAnnotationSaveAvailability();
  }
}

el("refresh-window-regions").addEventListener("click", async () => {
  const captureId = state.capture?.id;
  if (!captureId || state.liveView) return;
  const button = el("refresh-window-regions");
  button.disabled = true;
  try {
    const regions = await request(`/api/captures/${encodeURIComponent(captureId)}/regions`);
    if (state.capture?.id !== captureId) return;
    state.regions = regions;
    renderRegions();
    renderBatchBoxes();
    state.selectionMergeRegions = state.selection
      ? boxesCrossedBySelection(state.selection, state.editRegion?.id)
      : [];
    updateSelectionMergeControls();
    drawOverlays();
    el("batch-box-status").textContent = "Saved signals refreshed. Your draft is unchanged. Undo and restage edits or merges based on older signal versions before saving.";
  } catch (error) {
    el("batch-box-status").textContent = `Could not refresh saved signals · ${error.message}. Your draft is unchanged.`;
  } finally {
    button.disabled = false;
  }
});

function reviewBatchBox(index) {
  const box = state.batchBoxes[index];
  if (!box || !state.capture) return;
  saveSelectedBatchLabelDraft();
  state.batchBoxMode = false;
  beginNewRegion();
  state.batchReviewBox = box;
  updateSingleBoxButton();
  renderBatchBoxes();
  state.selection = { ...box };
  state.selectionMergeRegions = boxesCrossedBySelection(state.selection);
  updateSelectionMergeControls();
  state.classifierProposal = null;
  restoreBatchLabelDraft(box);
  reprojectSelection();
  updateCarrierFromSelection(state.selection);
  openInspector("label");
  const suggested = box.result?.method || box.result?.family;
  el("selection-summary").textContent = `Signal ${index + 1} · ${suggested ? `suggestion: ${suggested}` : "label this signal"}`;
  el("selection-details").textContent = `Draft · ${(box.sample_start / state.capture.sample_rate).toFixed(2)}–${((box.sample_start + box.sample_count) / state.capture.sample_rate).toFixed(2)} s · ${box.frequency_low_hz.toFixed(0)}–${box.frequency_high_hz.toFixed(0)} Hz. Drag inside to move or a corner to resize; its label stays with it.`;
  el("label-state").textContent = box.labelDraft?.mode_id || box.labelDraft?.family || box.labelDraft?.identity_unknown_confirmed
    ? "Label saved in this window draft."
    : suggested ? `Classifier suggests ${suggested}; choose a label to confirm it.` : "Choose a mode or leave unlabeled for now.";
  setRegionSaveDisabled(!state.selection || state.liveView || !state.capture);
  updateModelButton();
  updateSceneReviewAvailability();
  drawOverlays();
}

el("batch-box-toggle").addEventListener("click", () => {
  if (!state.capture || state.liveView || state.coverageOnly) return;
  saveSelectedBatchLabelDraft();
  if (state.batchBoxMode) {
    state.batchBoxMode = false;
  } else {
    if (state.drawingSignalBox) cancelSingleBoxDrawing("Single-box drawing stopped · multi-box drawing is active.");
    state.batchBoxMode = true;
    state.selection = null;
    state.classifierProposal = null;
    beginNewRegion();
    el("selection-summary").textContent = "Multi-box selection";
    el("selection-details").textContent = "Draw separate boxes around each signal. Each box will receive an independent provisional classification.";
    setRegionSaveDisabled(true);
    updateModelButton();
    updateSceneReviewAvailability();
  }
  updateSingleBoxButton();
  renderBatchBoxes();
  drawOverlays();
});

el("batch-box-clear").addEventListener("click", () => {
  if (state.batchBoxes.length || state.windowMerges.length) state.batchBoxUndo.push({type: "restore-all", boxes: [...state.batchBoxes], merges: [...state.windowMerges]});
  if (state.batchReviewBox) clearCurrentSelection("All draft signals removed.");
  state.batchBoxes = [];
  state.windowMerges = [];
  state.windowIdentityLinks = [];
  state.identityLinkSelection.clear();
  state.batchReviewBox = null;
  state.batchBoxMode = false;
  renderBatchBoxes();
  drawOverlays();
});

function undoBatchBoxChange() {
  const change = state.batchBoxUndo.pop();
  if (!change) return false;
  if (change.type === "add") {
    state.batchBoxes = state.batchBoxes.filter((box) => box !== change.box);
    if (state.batchReviewBox === change.box) state.batchReviewBox = null;
  } else if (change.type === "remove") {
    state.batchBoxes.splice(Math.max(0, Math.min(change.index, state.batchBoxes.length)), 0, change.box);
  } else if (change.type === "remove-merge") {
    state.batchBoxes.splice(Math.max(0, Math.min(change.index, state.batchBoxes.length)), 0, change.box);
    if (change.merge) state.windowMerges.push(change.merge);
  } else if (change.type === "restore-all") {
    state.batchBoxes = change.boxes;
    if (change.merges) state.windowMerges = change.merges;
  } else if (change.type === "delete-region") {
    state.windowRegionDeletes.delete(change.regionId);
    if (change.previousEdit) state.windowRegionEdits.set(change.regionId, change.previousEdit);
  } else if (change.type === "restore-region-edit") {
    if (change.previousDraft) state.windowRegionEdits.set(change.regionId, change.previousDraft);
    else state.windowRegionEdits.delete(change.regionId);
    const region = state.regions.find((item) => item.id === change.regionId);
    if (region && state.editRegion?.id === change.regionId) {
      state.editRegion = null;
      selectSavedRegion(region);
    }
  } else if (change.type === "undo-merge") {
    state.windowMerges = change.previousMerges || state.windowMerges.filter((merge) => merge !== change.merge);
    state.batchBoxes = change.previousBoxes || state.batchBoxes.filter((box) => box !== change.box);
    for (const edit of change.priorSourceEdits || []) state.windowRegionEdits.set(edit.region_id, edit);
    state.batchReviewBox = change.previousSelection || null;
    state.editRegion = change.previousEditRegion || null;
    state.selection = change.previousGeometrySelection ? {...change.previousGeometrySelection} : null;
  } else if (change.type === "restore-box-geometry") {
    Object.assign(change.box, change.geometry);
    if (state.batchReviewBox === change.box) {
      state.selection = {...change.geometry};
      reprojectSelection();
    }
  } else if (change.type === "restore-box-label") {
    change.box.labelDraft = {...change.labelDraft};
    if (state.batchReviewBox === change.box) restoreBatchLabelDraft(change.box);
  }
  renderBatchBoxes();
  renderRegions();
  state.selectionMergeRegions = state.selection ? boxesCrossedBySelection(state.selection, state.editRegion?.id) : [];
  updateSelectionMergeControls();
  drawOverlays();
  updateAnnotationSaveAvailability();
  return true;
}

el("batch-box-classify").addEventListener("click", async () => {
  const capture = state.capture;
  if (!capture || !state.modelAvailable || !state.batchBoxes.length || state.batchBoxMode) return;
  state.batchClassifying = true;
  renderBatchBoxes();
  let savedCount = 0;
  let unknownCount = 0;
  try {
    for (const box of [...state.batchBoxes]) {
      try {
        box.engine = "method-model";
        box.result = await request(`/api/captures/${capture.id}/classify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sample_start: box.sample_start,
            sample_count: Math.min(1_000_000, box.sample_count),
            frequency_low_hz: box.frequency_low_hz,
            frequency_high_hz: box.frequency_high_hz,
            crop_count: Number(el("classifier-crop-count").value),
          }),
        });
        box.error = null;
        savedCount += 1;
        if (!box.result.family && !box.result.method) unknownCount += 1;
      } catch (error) {
        box.error = error.message;
      }
      if (state.capture !== capture) return;
      renderBatchBoxes();
    }
  } finally {
    state.batchClassifying = false;
    renderBatchBoxes();
    if (state.capture === capture && savedCount) {
      el("batch-box-status").textContent = `${savedCount} suggestion${savedCount === 1 ? "" : "s"} ready to review${unknownCount ? ` · ${unknownCount} have no confident identity` : ""}. Choose labels, then save the window changes together.${state.batchBoxes.some((box) => box.error) ? " Some suggestions failed; those boxes remain available for manual labeling." : ""}`;
    }
  }
});

function updateModelButton() {
  const tooShort = (state.selection?.sample_count ?? 0) < 1024;
  el("model-propose").disabled = !state.modelAvailable || !state.capture || !state.selection || state.liveView || tooShort;
  el("model-propose").title = tooShort ? "The classifier needs at least 1024 audio samples in this selection." : state.selectedTrack?.cv_proposal_id && !state.editRegion
    ? "Classify this CV box and save a pending region, including unknown identity when the model abstains."
    : "Suggest a modulation method for this selection.";
  el("model-propose").textContent = state.selectedTrack?.cv_proposal_id && !state.editRegion
    ? "Classify and save region" : "Suggest method";
  const cropCount = Number(el("classifier-crop-count").value);
  el("classifier-crop-count-value").value = String(cropCount);
  el("classifier-crop-count-value").textContent = String(cropCount);
  updateCvSimilarButton();
}

el("classifier-crop-count").addEventListener("input", () => {
  try { localStorage.setItem("sig-id-classifier-crop-count", el("classifier-crop-count").value); } catch {}
  updateModelButton();
});

el("live-cv-merge-overlap-value").value = `${liveCvMergeOverlap}%`;
el("live-cv-merge-overlap-value").textContent = `${liveCvMergeOverlap}%`;
el("live-cv-merge-overlap").addEventListener("input", () => {
  const value = el("live-cv-merge-overlap").value;
  el("live-cv-merge-overlap-value").value = `${value}%`;
  el("live-cv-merge-overlap-value").textContent = `${value}%`;
  try { localStorage.setItem("sig-id-live-cv-merge-overlap", value); } catch {}
});

el("model-propose").addEventListener("click", async () => {
  if (!state.capture || !state.selection || state.liveView || !state.modelAvailable) return;
  const selection = state.selection;
  const sourceCapture = state.capture;
  state.classifierProposal = null;
  clearDecoderRecommendations();
  const button = el("model-propose");
  button.disabled = true;
  button.textContent = "Analyzing…";
  el("label-state").textContent = "Calculating a provisional model proposal…";
  try {
    const result = await request(`/api/captures/${state.capture.id}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sample_start: selection.sample_start,
        sample_count: Math.min(1_000_000, selection.sample_count),
        frequency_low_hz: selection.frequency_low_hz,
        frequency_high_hz: selection.frequency_high_hz,
        crop_count: Number(el("classifier-crop-count").value),
      }),
    });
    if (state.capture !== sourceCapture || state.selection !== selection) return;
    state.classifierProposal = {
      capture_id: sourceCapture.id,
      selection,
      result: { ...result, selection_sample_count: selection.sample_count },
    };
    if (!result.family && !result.method) {
      clearDecoderRecommendations();
      el("label-state").textContent = result.novelty_rejected
        ? `Model abstained · outside trained modulation range (distance ${result.novelty_distance.toFixed(2)} > ${result.novelty_threshold.toFixed(2)})`
        : `Model abstained · ${result.requested_crop_count > 1 ? `no majority across ${result.crop_count} crops · ` : ""}top guess ${result.top_candidate} at ${(100 * result.confidence).toFixed(1)}%`;
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
      el("label-state").textContent = `Proposed ${result.method || result.family} · ${result.requested_crop_count > 1 ? `${result.consensus_count}/${result.crop_count} crops agree` : "single crop"} · review then save to accept`;
      await updateDecoderRecommendations(result, selection);
    }
    if (state.capture !== sourceCapture || state.selection !== selection) return;
    if (state.selectedTrack?.cv_proposal_id && !state.editRegion) {
      const saved = await savePendingClassifierRegion(sourceCapture, selection, state.selectedTrack, result);
      if (state.capture !== sourceCapture || state.selection !== selection) return;
      state.regions = await request(`/api/captures/${sourceCapture.id}/regions`);
      renderRegions();
      await refreshSceneCoverage(sourceCapture.id);
      selectSavedRegion(saved);
      const sameBounds = saved.sample_start === selection.sample_start
        && saved.sample_count === selection.sample_count
        && saved.frequency_low_hz === selection.frequency_low_hz
        && saved.frequency_high_hz === selection.frequency_high_hz;
      el("label-state").textContent = !sameBounds
        ? "This CV proposal already has a saved region with different bounds. Adjust that region before reclassifying."
        : saved.review_state === "pending"
          ? result.family || result.method
            ? "Classifier label saved as pending. Review it, then save the revision to accept."
            : "Classifier abstained. The box is saved with unknown identity for manual review."
          : "This selection already has a reviewed region; no duplicate was added.";
    }
  } catch (error) {
    el("label-state").textContent = `Model proposal failed · ${error.message}`;
  } finally {
    updateModelButton();
  }
});

let decoderRecommendationRequest = 0;
function clearDecoderRecommendations(message = "Run the classifier on a selected signal to see compatible modes.") {
  decoderRecommendationRequest += 1;
  const panel = el("decoder-recommendations");
  if (!panel) return;
  panel.hidden = true;
  el("decoder-recommendation-status").textContent = message;
  el("decoder-recommendation-list").replaceChildren();
}

async function updateDecoderRecommendations(result, selection) {
  const requestId = ++decoderRecommendationRequest;
  const panel = el("decoder-recommendations");
  const status = el("decoder-recommendation-status");
  const list = el("decoder-recommendation-list");
  panel.hidden = false;
  list.replaceChildren();
  if (!result.method) {
    status.textContent = "No specific modulation method was proposed, so no decoder modes are suggested.";
    return;
  }
  if (!state.fldigiModes?.length) {
    status.textContent = "Connect to Fldigi to compare classifier results with its selectable mode catalog.";
    return;
  }
  status.textContent = `Checking selectable Fldigi modes compatible with ${result.method}…`;
  try {
    const response = await request(`/api/modes?q=${encodeURIComponent(result.method)}&limit=50`);
    if (requestId !== decoderRecommendationRequest || state.selection !== selection) return;
    const method = result.method.trim().toLocaleLowerCase();
    const catalog = new Map(state.fldigiModes.map((name) => [name.toLocaleLowerCase(), name]));
    const candidates = [...new Set(response.modes
      .filter((item) => item.method?.trim().toLocaleLowerCase() === method
        || item.display_name?.trim().toLocaleLowerCase() === method
        || item.variant?.trim().toLocaleLowerCase() === method)
      .map((item) => catalog.get(item.display_name.toLocaleLowerCase()))
      .filter(Boolean))];
    if (!candidates.length) {
      status.textContent = `No selectable Fldigi modes in the current catalog match ${result.method}.`;
      return;
    }
    status.textContent = `${candidates.length} selectable mode${candidates.length === 1 ? "" : "s"} match the proposed ${result.method} modulation:`;
    for (const name of candidates) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Stage ${name}`;
      button.addEventListener("click", () => {
        if (state.selection !== selection || !state.fldigiModes?.includes(name)) return;
        el("fldigi-mode").value = name;
        updateDecoderSelectionControls();
        el("fldigi-status").textContent = `${name} staged from classifier suggestion · select receive mode to apply it`;
        openInspector("decode");
      });
      list.append(button);
    }
  } catch (error) {
    if (requestId !== decoderRecommendationRequest || state.selection !== selection) return;
    status.textContent = `Could not load compatible mode suggestions · ${error.message}`;
  }
}

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
  if ((state.batchBoxes.length || state.windowRegionEdits.size || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length)
    && !await resolveWindowDraftsBeforeSwitch("starting live capture")) return;
  try {
    const device = el("audio-device").value;
    const captureMode = el("live-capture-mode").value;
    const noiseIntervalSeconds = Number(el("noise-interval-minutes").value) * 60;
    const started = await request("/api/live/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_id: device === "" ? null : Number(device),
        sample_rate: Number(el("audio-rate").value), channels: 1,
        max_duration_seconds: captureMode === "unattended" ? 28800 : 600,
        capture_mode: captureMode, noise_interval_seconds: noiseIntervalSeconds,
      }),
    });
    state.captureMode = captureMode;
    el("live-start").disabled = true;
    el("live-stop").disabled = false;
    el("live-stop").textContent = captureMode === "record" ? "Stop and review" : "Stop live scan";
    clearDemodExports();
    state.capture = null;
    state.cvDetections = [];
    state.cvDismissedCount = 0;
    state.cvRejectedProposals = [];
    renderCvDetections();
    renderCvDismissedProposals();
    updateCvButton();
    state.liveView = true;
    state.liveSessionId = started.session_id;
    state.liveSampleRate = started.sample_rate;
    state.liveWindowStartSample = null;
    state.liveWindowEndSample = null;
    setInspectorMode("classify");
    openInspector("label");
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
    el("capture-title").textContent = captureMode === "unattended" ? "Unattended signal monitor" : "Live passband";
    const modeDescription = captureMode === "record" ? "recording full audio" : captureMode === "scan_only" ? "audio stays in memory" : "only selected clips are saved";
    el("capture-meta").textContent = `${started.sample_rate.toLocaleString()} samples/s · ${started.device_name} · ${modeDescription}`;
    el("time-start").textContent = "Now";
    el("time-end").textContent = "−30 s";
    el("audio-player").removeAttribute("src");
    el("waterfall-state").textContent = "Waiting for audio frames…";
    el("track-status").textContent = "Waiting for enough samples to analyze…";
    setRegionSaveDisabled(true);
    updateModelButton();
    renderTracks();
    el("live-status").textContent = captureMode === "record"
      ? `Recording · ${started.device_name} · ${started.sample_rate.toLocaleString()} Hz`
      : captureMode === "scan_only"
        ? `Live scan only · no recording is being saved · ${started.device_name}`
        : `Unattended monitoring · selected clips go to the review queue · ${started.device_name}`;
    renderRegions();
    renderDecodeRuns();
    const generation = state.liveGeneration = (state.liveGeneration || 0) + 1;
    liveDisplay = new LiveWaterfall(60);
    startSpectrumStream(generation, started.session_id);
    let pending = false;
    let lastLiveCvAt = 0;
    livePoll = setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const status = await request("/api/live");
        if (generation !== state.liveGeneration || !state.liveView) return;
        if (status.active) {
          const modeLabel = state.captureMode === "record" ? "Recording" : state.captureMode === "scan_only" ? "Live scan only · no audio saved" : `Unattended · ${status.saved_review_clips || 0} clips queued`;
          el("live-status").textContent = state.frozen ? `${modeLabel} · view frozen` : `${modeLabel} · ${status.frames_written.toLocaleString()} samples analyzed`;
        }
        const cvEnabled = el("live-cv-enabled").checked;
        const classifierEnabled = el("live-classifier-enabled").checked;
        const autoTrial = el("auto-trial").checked && classifierEnabled;
        const useHeuristicTracks = classifierEnabled && (!cvEnabled || autoTrial);
        const classifierCropCount = Number(el("classifier-crop-count").value) || 1;
        const analysis = await request(`/api/live/analysis?fft_size=4096&hop_size=512&max_rows=640&include_spectrum=false&classifier_crop_count=${classifierCropCount}&include_tracks=${useHeuristicTracks}&include_classifier=${useHeuristicTracks}&auto_trial=${autoTrial && useHeuristicTracks}`);
        if (generation !== state.liveGeneration || !state.liveView) return;
        state.liveWindowStartSample = Number.isFinite(analysis.sample_offset) ? analysis.sample_offset : null;
        state.liveWindowEndSample = Number.isFinite(analysis.sample_offset) && Number.isFinite(analysis.window_sample_count)
          ? analysis.sample_offset + analysis.window_sample_count : null;
        syncLiveAnnotationSaveButton();
        if (cvEnabled && state.cvAvailable && Date.now() - lastLiveCvAt >= 2500) {
          lastLiveCvAt = Date.now();
          try {
            const cv = await request(`/api/live/cv-detections?session_id=${encodeURIComponent(started.session_id)}&include_classifier=${classifierEnabled}`);
            if (generation === state.liveGeneration && state.liveView) {
              mergeLiveCvDetections(cv.detections);
              if (!classifierEnabled) {
                for (const detection of state.cvDetections) delete detection.classifier_proposal;
              }
              renderCvDetections();
              const visibleLiveDetections = visibleCvDetections().length;
              const visibleScannedDetections = (cv.detections || []).filter((item) => !frequencyNotchMatches(item)).length;
              el("cv-status").textContent = cv.warming_up
                ? `Live CV warming up · ${cv.observed_seconds.toFixed(2)} s of audio`
                : `Live CV · ${visibleLiveDetections} visible boxes${state.cvDetections.length > visibleLiveDetections ? ` · ${state.cvDetections.length - visibleLiveDetections} suppressed by notches` : ""} · ${visibleScannedDetections} in latest scan · ${classifierEnabled ? (cv.classifier_available ? "classifier proposals attached" : "classifier unavailable") : "classifier paused"} · proposals are provisional`;
            }
          } catch (error) {
            if (generation === state.liveGeneration) el("cv-status").textContent = `Live CV unavailable · ${error.message}`;
          }
        } else if (!cvEnabled) {
          state.cvDetections = [];
          renderCvDetections();
          el("cv-status").textContent = "CV paused · no CV boxes are being generated";
        } else if (!state.cvAvailable) {
          state.cvDetections = [];
          renderCvDetections();
          el("cv-status").textContent = "CV is selected but unavailable · check the detector model status";
        }
        if (generation !== state.liveGeneration || !state.liveView) return;
        if (analysis.automatic_trial?.run_id) watchDecoderTrial(analysis.automatic_trial);
        if (state.frozen) return;
        if (analysis.warming_up) {
          el("waterfall-state").textContent = `Collecting analysis window · ${analysis.window_sample_count.toLocaleString()} samples`;
          return;
        }
        const showHeuristicTracks = !cvEnabled && classifierEnabled;
        state.tracks = showHeuristicTracks ? analysis.tracks : [];
        state.liveTrackModeMessage = cvEnabled
          ? (state.cvAvailable
            ? "CV boxes are active above; heuristic track boxes are hidden. Turn CV off to inspect classifier-only track candidates."
            : "CV is selected but unavailable. Turn CV off to inspect classifier-only track candidates.")
          : classifierEnabled
            ? null
            : "Candidate analysis is paused. Turn on CV signal boxes or classifier suggestions to show candidates.";
        el("track-status").textContent = cvEnabled
          ? "CV detection mode · classifier suggestions are attached to CV boxes when enabled"
          : classifierEnabled
            ? "Classifier-only mode · heuristic candidate bounds are provisional"
            : "Candidate analysis paused";
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
    state.liveSessionId = null;
    if (result.capture) {
      state.capture = result.capture;
      await loadCapture();
      el("live-status").textContent = result.truncated_reason
        ? `Saved partial live capture · ${result.truncated_reason} · ${result.frames_written.toLocaleString()} samples`
        : `Saved live capture · ${result.frames_written.toLocaleString()} samples`;
    } else {
      state.liveView = false;
      state.capture = null;
      state.loadedCapture = null;
      resetOutputSession();
      state.waterfall = null;
      state.view = null;
      state.tracks = [];
      state.regions = [];
      el("workspace").classList.add("hidden");
      el("empty-state").classList.remove("hidden");
      el("live-status").textContent = state.captureMode === "unattended"
        ? `Monitoring stopped · ${result.saved_review_clips} selected 10-second clips are in the review queue`
        : "Live scan stopped · no recording was saved";
      if (state.captureMode === "unattended") refreshLiveReviewQueue();
    }
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
    if (!state.liveView) {
      el("live-stop").textContent = "Stop and review";
      state.captureMode = null;
      updateLiveCaptureModeControls();
    }
  }
});

function updateLiveCaptureModeControls() {
  const unattended = el("live-capture-mode").value === "unattended";
  el("noise-interval-wrap").hidden = !unattended;
  const mode = el("live-capture-mode").value;
  el("live-start").textContent = mode === "scan_only"
    ? "Start scan without saving"
    : mode === "unattended" ? "Start unattended monitor" : "Start live capture";
}
el("live-capture-mode").addEventListener("change", updateLiveCaptureModeControls);
updateLiveCaptureModeControls();

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
    const detail = typeof body.detail === "string"
      ? body.detail
      : Array.isArray(body.detail)
        ? body.detail.map((issue) => {
          const location = Array.isArray(issue.loc) ? issue.loc.join(" → ") : "request";
          return `${location}: ${issue.msg || issue.type || "invalid value"}`;
        }).join(" · ")
        : body.detail && typeof body.detail === "object"
          ? JSON.stringify(body.detail)
          : `Request failed (${response.status})`;
    const error = new Error(detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function ensureWindowSaveCapabilities({updates, deletions, merges, identityLinks = []}) {
  if (!updates.length && !deletions.length && !merges.length && !identityLinks.length) return;
  const schema = await request("/openapi.json");
  const properties = schema.components?.schemas?.WindowAnnotationBatchCreate?.properties || {};
  const missing = [
    updates.length && !properties.updates && "saved-signal edits",
    deletions.length && !properties.deletions && "signal removals",
    merges.length && !properties.merges && "merges",
    identityLinks.length && !properties.identity_links && "signal identity links",
  ].filter(Boolean);
  if (missing.length) {
    throw new Error(`This running workbench server is older than the annotation editor and cannot save ${missing.join(", ")}. Your draft is still here. Restart the workbench from the current checkout before saving these changes.`);
  }
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
  const support = [
    mode.status || "curated",
    mode.detector_status && `detector: ${mode.detector_status}`,
    mode.decoder_status && `decoder: ${mode.decoder_status}`,
  ].filter(Boolean).join(" · ");
  el("mode-resolution").textContent = `${mode.display_name} → ${mode.family} / ${mode.method} / ${mode.protocol} / ${mode.variant} · ${support} · registry ${mode.registry_version}`;
  el("mode-results").replaceChildren();
  state.applyingMode = false;
  el("batch-unknown-confirmed").checked = false;
  if (state.batchReviewBox) {
    saveSelectedBatchLabelDraft();
    renderBatchBoxes();
    drawOverlays();
  } else saveSelectedExistingRegionDraft();
}

el("batch-unknown-confirmed").addEventListener("change", () => {
  if (el("batch-unknown-confirmed").checked) {
    state.labelModeId = null;
    el("mode-search").value = "";
    for (const field of ["family", "method", "protocol", "variant"]) el(field).value = "";
  }
  saveSelectedBatchLabelDraft();
  el("mode-resolution").textContent = el("batch-unknown-confirmed").checked
    ? "Confirmed signal, identity unknown."
    : "Choose a mode, enter a partial label, or leave this signal unlabeled.";
  renderBatchBoxes();
  drawOverlays();
});

el("mode-search").addEventListener("input", async () => {
  const query = el("mode-search").value.trim();
  const replacingCanonicalMode = Boolean(state.labelModeId);
  clearModeSelection(false);
  if (replacingCanonicalMode) {
    for (const field of ["family", "method", "protocol", "variant"]) el(field).value = "";
  }
  el("batch-unknown-confirmed").checked = false;
  saveSelectedBatchLabelDraft();
  const requestId = ++state.modeSearchRequest;
  const results = el("mode-results");
  results.replaceChildren();
  if (query.length < 2) return;
  try {
    const result = await request(`/api/modes?q=${encodeURIComponent(query)}`);
    if (requestId !== state.modeSearchRequest) return;
    state.modeSearchResults = result.modes;
    for (const mode of result.modes) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-result";
      button.setAttribute("role", "option");
      const title = document.createElement("strong");
      title.textContent = mode.display_name;
      const taxonomy = document.createElement("span");
      taxonomy.textContent = [mode.family, mode.method, mode.protocol, mode.variant].filter(Boolean).join(" / ");
      const capability = document.createElement("small");
      capability.textContent = [mode.status, mode.detector_status && `ID: ${mode.detector_status}`, mode.decoder_status && `decode: ${mode.decoder_status}`].filter(Boolean).join(" · ");
      button.append(title, taxonomy, capability);
      button.addEventListener("click", () => applyModeSelection(mode));
      results.append(button);
    }
    el("mode-resolution").textContent = result.modes.length
      ? `${result.modes.length} canonical match${result.modes.length === 1 ? "" : "es"} · ${result.modes.slice(0, 3).map((mode) => {
        const support = [mode.status || "registry mode", mode.detector_status && `ID ${mode.detector_status}`, mode.decoder_status && `decode ${mode.decoder_status}`].filter(Boolean).join(", ");
        return `${mode.display_name}: ${support}`;
      }).join(" · ")} · choose a result to fill the taxonomy.`
      : "No canonical match · keep a partial or unknown label if appropriate.";
  } catch (error) {
    if (requestId === state.modeSearchRequest) el("mode-resolution").textContent = `Mode lookup unavailable: ${error.message}`;
  }
});

for (const field of ["family", "method", "protocol", "variant"]) {
  el(field).addEventListener("input", () => {
    if (state.applyingMode) return;
    if (state.labelModeId) {
      state.labelModeId = null;
      el("mode-search").value = "";
      for (const other of ["family", "method", "protocol", "variant"]) {
        if (other !== field) el(other).value = "";
      }
    }
    if (state.batchReviewBox) {
      el("batch-unknown-confirmed").checked = false;
      saveSelectedBatchLabelDraft();
      drawOverlays();
    } else saveSelectedExistingRegionDraft();
  });
}
el("notes").addEventListener("input", () => {
  saveSelectedBatchLabelDraft();
});

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

function setCvCoverageQueueStatus(message, showLabel = true) {
  el("cv-coverage-queue-status").textContent = message;
  const labelStatus = el("cv-coverage-queue-label-status");
  labelStatus.textContent = message;
  labelStatus.hidden = !showLabel;
}

function chooseAnnotationDraftAction(message) {
  const dialog = el("annotation-draft-dialog");
  el("annotation-draft-message").textContent = message;
  dialog.returnValue = "";
  const choice = new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue || "stay"), {once: true});
  });
  dialog.showModal();
  return choice;
}

async function resolveWindowDraftsBeforeSwitch(context = "switching recordings") {
  saveSelectedBatchLabelDraft();
  if (!state.liveView) stageCurrentSelectionAsWindowDraft();
  const hasDraftChanges = state.batchBoxes.length || state.windowRegionEdits.size || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length;
  if (!hasDraftChanges) return true;
  const action = await chooseAnnotationDraftAction(`This 10-second window has unsaved changes. Save or discard them before ${context}?`);
  if (action === "stay") return false;
  if (action === "save") {
    try {
      await saveWindowAnnotationDrafts();
      return true;
    } catch (error) {
      showError(`Could not save this window, so it stayed open · ${error.message}`);
      return false;
    }
  }
  if (action !== "discard") return false;
  state.batchBoxes = [];
  state.batchBoxUndo = [];
  state.batchReviewBox = null;
  state.windowRegionEdits.clear();
  state.windowRegionDeletes.clear();
  state.windowMerges = [];
  state.windowIdentityLinks = [];
  state.identityLinkSelection.clear();
  state.batchSaveKey = null;
  state.batchSaveDigest = null;
  renderBatchBoxes();
  drawOverlays();
  return true;
}

async function loadCapture({windowIndex = 0, focusedWindow = false} = {}) {
  const incomingCapture = state.capture;
  const previousCapture = state.loadedCapture;
  if (previousCapture && incomingCapture?.id !== previousCapture.id) {
    state.capture = previousCapture;
    if (!await resolveWindowDraftsBeforeSwitch("switching recordings")) return false;
    state.capture = incomingCapture;
  }
  state.loadedCapture = incomingCapture;
  state.annotationWindowIndex = windowIndex;
  stopSelectedAudioPlayback();
  state.closeInspectorAfterSave = false;
  state.returnInspectorTabAfterSave = null;
  state.markCaptureCompleteAfterSave = false;
  state.mergeRegionIds.clear();
  state.excludedMergeCandidates.clear();
  state.selectionMergeRegions = [];
  state.geometryMergeRegions = [];
  const generation = state.captureGeneration = (state.captureGeneration || 0) + 1;
  state.batchBoxMode = false;
  state.batchBoxes = [];
  state.batchBoxUndo = [];
  state.windowRegionEdits.clear();
  state.windowRegionDeletes.clear();
  state.windowMerges = [];
  state.windowIdentityLinks = [];
  state.identityLinkSelection.clear();
  state.batchSaveKey = null;
  state.batchSaveDigest = null;
  state.batchClassifying = false;
  renderBatchBoxes();
  clearDemodExports();
  state.liveView = false;
  state.liveSessionId = null;
  resetOutputSession();
  state.coverageRecords = [];
  state.cvCoverageQueue = null;
  setCvCoverageQueueStatus("Review the recording in 10-second windows. Complete empty windows count as negative examples.", false);
  state.coverageEdit = null;
  state.coverageOnly = false;
  state.coverageWindow = null;
  state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
  state.coverageMaskMode = null;
  state.adjustingCandidate = null;
  state.drawingSignalBox = false;
  updateSingleBoxButton();
  el("coverage-panel").hidden = true;
  state.editRegion = null;
  state.selectionMergeRegions = [];
  updateSelectionMergeControls();
  state.geometryMergeRegions = [];
  el("merge-redrawn-regions").hidden = true;
  el("geometry-merge-status").hidden = true;
  el("geometry-merge-status").textContent = "";
  state.classifierProposal = null;
  clearDecoderRecommendations();
  el("start-scene-review").disabled = true;
  updateCoverageMaskControls();
  state.waterfall = null;
  updateSceneReviewAvailability();
  baseCanvas.getContext("2d").clearRect(0, 0, baseCanvas.width, baseCanvas.height);
  overlayCanvas.getContext("2d").clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  el("signal-labels")?.replaceChildren();
  state.view = null;
  state.selection = null;
  beginNewRegion();
  el("selection-summary").textContent = "No region selected";
  el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
  el("inspector-context").textContent = "Select a signal on the waterfall to inspect it.";
  el("box-draw-status").textContent = "Drag on the waterfall to select a box, then save it as a region.";
  state.cvDetections = [];
  state.cvDismissedCount = 0;
  state.cvRejectedProposals = [];
  renderCvDismissedProposals();
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
  const queuedReviewCapture = Boolean(c.source_metadata?.auto_review_queue && !c.source_metadata?.auto_review_queue_reviewed_utc);
  el("save-annotations-complete").hidden = !queuedReviewCapture;
  el("save-annotations-complete").title = queuedReviewCapture
    ? "Save this annotation, record complete detection coverage for this 10-second review window, and return to the review queue."
    : "Available for samples in the live review queue.";
  el("save-annotations-complete").textContent = state.pendingCaptureCompletion === c.id
    ? "Retry finish"
    : "Save and finish";
  renderBatchBoxes();
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
  const spectrumParams = new URLSearchParams({fft_size: "4096", hop_size: "512", max_rows: focusedWindow ? "8192" : "640"});
  if (focusedWindow) {
    const bounds = annotationWindowBounds(c);
    spectrumParams.set("sample_start", String(bounds.startSample));
    spectrumParams.set("window_sample_count", String(bounds.sampleCount));
  }
  const spectrum = await request(`/api/captures/${c.id}/waterfall?${spectrumParams}`);
  if (generation !== state.captureGeneration || state.liveView) return;
  state.waterfall = spectrum;
  drawWaterfall();
  updateAnnotationWindowView();
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
  return true;
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
      el("coverage-list").textContent = `Window reviews unavailable · ${error.message}`;
    }
    return null;
  }
}

function updateSceneReviewAvailability() {
  const selection = state.selection;
  const valid = Boolean(state.capture && !state.liveView && !state.coverageOnly && selection && Number.isInteger(selection.sample_start) && Number.isInteger(selection.sample_count) && selection.sample_count > 0 && !state.adjustingGeometry && !state.splitMode);
  el("start-scene-review").disabled = !valid;
  const fullReviewWindowSamples = state.capture ? state.capture.sample_rate * state.cvReviewWindowSeconds : Infinity;
  const canReviewCoverage = Boolean(state.capture && state.waterfall && !state.liveView
    && state.capture.sample_count >= fullReviewWindowSamples
    && state.capture.sample_rate / 2 >= state.cvReviewMaxFrequencyHz);
  el("start-cv-coverage-queue").disabled = !canReviewCoverage;
}

function cvReviewWindows(capture) {
  const windowSamples = capture.sample_rate * state.cvReviewWindowSeconds;
  const windows = [];
  for (let start = 0; start + windowSamples <= capture.sample_count; start += windowSamples) {
    windows.push({ sample_start: start, sample_count: windowSamples });
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
    setCvCoverageQueueStatus(message);
    el("coverage-status").textContent = message;
    return;
  }
  const progress = `${queue.alreadyComplete + queue.visited.size + 1}/${queue.totalWindows} accounted for · ${queue.completeCount} complete, ${queue.partialCount} partial/uncertain this pass`;
  setCvCoverageQueueStatus(progress);
  showCvReviewWindow(next, progress);
}

async function startCvCoverageQueue() {
  const capture = state.capture;
  if (!capture || !state.waterfall || state.liveView || capture.sample_rate / 2 < state.cvReviewMaxFrequencyHz) return;
  const records = await refreshSceneCoverage(capture.id);
  if (state.capture !== capture) return;
  if (!records) {
    setCvCoverageQueueStatus("Could not load window reviews; retry after checking the workbench connection.");
    return;
  }
  state.coverageRecords = records;
  const windows = cvReviewWindows(capture);
  const pending = windows.filter((window) => !hasCompleteCvCoverage(window));
  if (!pending.length) {
    state.cvCoverageQueue = null;
    setCvCoverageQueueStatus(`All ${windows.length} CV windows have fresh complete detection coverage.`);
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
  if (!state.cvCoverageQueue) el("cv-coverage-queue-label-status").hidden = true;
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
  setRegionSaveDisabled(true);
  const selection = state.selection;
  const start = selection.sample_start / state.capture.sample_rate;
  const end = (selection.sample_start + selection.sample_count) / state.capture.sample_rate;
  const low = selection.frequency_low_hz;
  const high = selection.frequency_high_hz;
  el("selection-summary").textContent = "Window coverage review";
  el("selection-details").textContent = `Review this exact window · ${start.toFixed(2)}–${end.toFixed(2)} s · ${Number.isFinite(low) && Number.isFinite(high) ? `${low.toFixed(0)}–${high.toFixed(0)} Hz` : "frequency scope unknown"}`;
  el("coverage-status").textContent = "Check for all in-scope signals; choose Partial / uncertain when the window does not support a complete claim.";
  openInspector("signals");
}

function exitSceneCoverageReview() {
  state.coverageEdit = null;
  state.coverageOnly = false;
  state.coverageWindow = null;
  state.coverageMasks = { ignore_regions: [], uncertain_regions: [] };
  state.coverageMaskMode = null;
  el("coverage-panel").hidden = true;
  el("save-coverage").textContent = "Record reviewed window";
  state.selection = null;
  beginNewRegion();
  el("selection-summary").textContent = "No region selected";
  el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
  drawOverlays();
  setRegionSaveDisabled(!state.selection || !state.capture || state.liveView || Boolean(state.editRegion?.lineage_state === "superseded"));
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
  if (!state.coverageRecords.length) list.textContent = "No window reviews for this capture yet.";
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
    ? "This window review is stale because overlapping annotations changed. Review this window again before saving."
    : `Reviewing ${record.state} scene window · annotation digest ${record.annotation_set_digest.slice(0, 12)}…`;
  setRegionSaveDisabled(true);
  el("save-region").textContent = "Review window selected";
  updateCarrierFromSelection(state.selection);
  reprojectSelection();
  state.coverageWindow = { ...state.selection };
  el("selection-summary").textContent = "Window review";
  el("selection-details").textContent = `Samples ${record.sample_start.toLocaleString()}–${(record.sample_start + record.sample_count).toLocaleString()} · ${record.task} review`;
  openInspector("signals");
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
    const detectorLink = run.detector_evidence?.proposal_id ? ` · CV proposal ${run.detector_evidence.proposal_id.slice(-8)}` : "";
    button.textContent = `${run.current_mode || "Fldigi"} · ${run.status}${carrier}${interval}${regionLink}${classifierLink}${detectorLink}`;
    button.addEventListener("click", () => {
      showOutput("text");
      state.textOutputContext = `Saved run ${run.id.slice(0,8)}${run.region_id ? ` · region ${run.region_id.slice(0,8)}` : ""}${run.classifier_evidence?.model_sha256 ? ` · classifier ${run.classifier_evidence.model_sha256.slice(0,8)}` : ""}${run.detector_evidence?.proposal_id ? ` · CV proposal ${run.detector_evidence.proposal_id.slice(-8)}` : ""} · ${run.output_scope} · ${run.association}`;
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
  for (const track of visibleTracks()) {
    const bounds = overlayBounds(track.sample_start, track.sample_count, track.frequency_low_hz, track.frequency_high_hz);
    if (!bounds) continue;
    const candidate = state.selectionMergeRegions.some((item) => item.kind === "track" && item.track === track);
    const excluded = candidate && state.excludedMergeCandidates.has(`track:${heuristicCandidateId(track)}`);
    const pending = pendingWindowMergeSources().tracks.has(heuristicCandidateId(track));
    ctx.strokeStyle = candidate ? excluded ? "#727784" : "#ffe27a"
      : track.track_state === "persistent" ? "#52d6d0" : "#f1be69";
    ctx.globalAlpha = excluded || pending ? 0.45 : 1;
    ctx.lineWidth = candidate ? 3 : 1.5;
    ctx.setLineDash(pending ? [1, 4] : [5, 3]);
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
    ctx.globalAlpha = 1;
  }
  for (const detection of visibleCvDetections()) {
    const bounds = overlayBounds(detection.sample_start, detection.sample_count,
      detection.frequency_low_hz, detection.frequency_high_hz);
    if (!bounds) continue;
    const pendingMerge = pendingWindowMergeSources().proposals.has(detection.cv_proposal_id || detection.proposal_id);
    const candidate = state.selectionMergeRegions.some((item) => item.kind === "cv" && item.detection === detection);
    const excluded = candidate && state.excludedMergeCandidates.has(`cv:${detection.cv_proposal_id || detection.proposal_id}`);
    ctx.strokeStyle = pendingMerge ? "#a18cb7" : "#bd8cff";
    if (candidate) ctx.strokeStyle = excluded ? "#727784" : "#ffe27a";
    ctx.lineWidth = candidate ? 3 : 2;
    ctx.globalAlpha = pendingMerge || excluded ? 0.45 : 1;
    ctx.setLineDash(pendingMerge ? [1, 4] : [2, 3]);
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
    ctx.globalAlpha = 1;
  }
  ctx.setLineDash([]);
  for (const region of state.regions) {
    if (region.lineage_state === "superseded" && !state.showSupersededRegions) continue;
    const editDraft = state.windowRegionEdits.get(region.id);
    const geometry = editDraft?.geometry || region;
    const low = geometry.frequency_low_hz ?? 0;
    const high = geometry.frequency_high_hz ?? data.sample_rate / 2;
    const bounds = overlayBounds(geometry.sample_start, geometry.sample_count, low, high);
    if (!bounds) continue;
    const removalDraft = state.windowRegionDeletes.has(region.id);
    const pendingMerge = pendingWindowMergeSources().regions.has(region.id);
    const candidate = state.selectionMergeRegions.some((item) => item.kind === "region" && item.region.id === region.id);
    const excluded = candidate && state.excludedMergeCandidates.has(`region:${region.id}`);
    ctx.strokeStyle = removalDraft ? "#f1be69" : editDraft ? "#6be3c8" : pendingMerge ? "#aeb7c2" : "#ffffff";
    if (candidate) ctx.strokeStyle = excluded ? "#727784" : "#ffe27a";
    ctx.globalAlpha = removalDraft || pendingMerge || excluded ? 0.45 : 1;
    ctx.lineWidth = editDraft || removalDraft || candidate ? 2.5 : 2;
    ctx.setLineDash(removalDraft ? [4, 4] : pendingMerge ? [1, 4] : []);
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
    ctx.globalAlpha = 1;
    ctx.setLineDash([]);
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
  state.batchBoxes.forEach((box) => {
    const bounds = overlayBounds(box.sample_start, box.sample_count, box.frequency_low_hz, box.frequency_high_hz);
    if (!bounds) return;
    const candidate = state.selectionMergeRegions.some((item) => item.kind === "draft" && item.box === box);
    const excluded = candidate && state.excludedMergeCandidates.has(`draft:${box.draftMergeId || box.client_id}`);
    ctx.strokeStyle = candidate ? excluded ? "#727784" : "#ffe27a"
      : state.batchReviewBox === box || (box.result && !box.error && (box.result.family || box.result.method)) ? "#6be3c8" : "#ff9f5a";
    ctx.globalAlpha = excluded ? 0.45 : 1;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(bounds.left, bounds.top, bounds.width, bounds.height);
    ctx.globalAlpha = 1;
  });
  if (state.selection) {
    const { x1, y1, x2, y2 } = state.selection;
    ctx.strokeStyle = "#72fff0";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2), Math.max(2, Math.abs(x2 - x1)), Math.max(2, Math.abs(y2 - y1)));
    ctx.setLineDash([]);
    if (!state.liveView && (state.editRegion || state.batchReviewBox || state.selectedTrack) && !state.coverageOnly) {
      ctx.fillStyle = "#72fff0";
      for (const [x, y] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
        ctx.fillRect(x - 4, y - 4, 8, 8);
        ctx.strokeStyle = "#07131c";
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 4, y - 4, 8, 8);
      }
    }
  }
  renderFrequencyNotches();
  renderSignalLabels();
}

function renderFrequencyNotches() {
  const container = el("frequency-notches");
  const strip = el("notch-place-strip");
  const clearButton = el("clear-notches");
  if (!container || !strip) return;
  container.replaceChildren();
  strip.classList.toggle("active", state.notchPlacement);
  strip.classList.toggle("dragging", Boolean(state.notchDrag));
  clearButton.hidden = !state.frequencyNotches.length;
  const data = state.waterfall;
  if (!data || !state.view) return;
  const bounds = waterfallViewBounds();
  const lowHz = data.frequency_hz[bounds.frequencyStart];
  const highHz = data.frequency_hz[bounds.frequencyEnd - 1];
  const width = container.clientWidth;
  if (!width || highHz <= lowHz) return;
  const xForFrequency = (frequency) => (frequency - lowHz) / (highHz - lowHz) * width;
  state.frequencyNotches.forEach((notch, index) => {
    if (notch.high_hz <= lowHz || notch.low_hz >= highHz) return;
    const left = Math.max(0, xForFrequency(notch.low_hz));
    const right = Math.min(width, xForFrequency(notch.high_hz));
    const color = notch.color || NOTCH_COLORS[index % NOTCH_COLORS.length];
    const band = document.createElement("div");
    band.className = "frequency-notch-band";
    band.style.left = `${left}px`;
    band.style.width = `${Math.max(2, right - left)}px`;
    band.style.setProperty("--notch-color", color);
    container.append(band);
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = "frequency-notch-marker";
    marker.style.left = `${(left + right) / 2}px`;
    marker.style.setProperty("--notch-color", color);
    marker.title = `Detection notch ${notch.low_hz.toFixed(1)}–${notch.high_hz.toFixed(1)} Hz. Click to remove.`;
    const triangle = document.createElement("span");
    triangle.className = "triangle";
    triangle.textContent = "▼";
    const range = document.createElement("span");
    range.textContent = `${Math.round(notch.low_hz)}–${Math.round(notch.high_hz)} Hz`;
    const remove = document.createElement("span");
    remove.className = "remove";
    remove.textContent = "×";
    marker.append(triangle, range, remove);
    marker.addEventListener("click", () => {
      state.frequencyNotches.splice(index, 1);
      saveFrequencyNotches();
    });
    container.append(marker);
  });
  if (state.notchDrag) {
    const preview = document.createElement("div");
    preview.className = "frequency-notch-band frequency-notch-preview";
    const start = Math.min(state.notchDrag.startX, state.notchDrag.currentX);
    const end = Math.max(state.notchDrag.startX, state.notchDrag.currentX);
    preview.style.left = `${start}px`;
    preview.style.width = `${Math.max(2, end - start)}px`;
    preview.style.setProperty("--notch-color", NOTCH_COLORS[state.frequencyNotches.length % NOTCH_COLORS.length]);
    container.append(preview);
  }
}

el("notch-mode").addEventListener("click", () => {
  state.notchPlacement = !state.notchPlacement;
  el("notch-mode").setAttribute("aria-pressed", String(state.notchPlacement));
  el("notch-mode").textContent = state.notchPlacement ? "Cancel notch" : "Place notch";
  renderFrequencyNotches();
});
el("clear-notches").addEventListener("click", () => {
  state.frequencyNotches = [];
  saveFrequencyNotches();
});
const notchPlaceStrip = el("notch-place-strip");
function notchPointerX(event) {
  const rect = notchPlaceStrip.getBoundingClientRect();
  return Math.max(0, Math.min(rect.width, event.clientX - rect.left));
}
function notchFrequencyAt(x) {
  const bounds = waterfallViewBounds();
  const data = state.waterfall;
  const lowHz = data.frequency_hz[bounds.frequencyStart];
  const highHz = data.frequency_hz[bounds.frequencyEnd - 1];
  return lowHz + x / Math.max(1, notchPlaceStrip.clientWidth) * (highHz - lowHz);
}
notchPlaceStrip.addEventListener("pointerdown", (event) => {
  if (!state.notchPlacement || !state.waterfall) return;
  event.preventDefault();
  event.stopPropagation();
  notchPlaceStrip.setPointerCapture(event.pointerId);
  const x = notchPointerX(event);
  state.notchDrag = {startX: x, currentX: x};
  renderFrequencyNotches();
});
notchPlaceStrip.addEventListener("pointermove", (event) => {
  if (!state.notchDrag) return;
  state.notchDrag.currentX = notchPointerX(event);
  renderFrequencyNotches();
});
notchPlaceStrip.addEventListener("pointerup", (event) => {
  if (!state.notchDrag || !state.waterfall) return;
  const drag = state.notchDrag;
  drag.currentX = notchPointerX(event);
  state.notchDrag = null;
  const first = notchFrequencyAt(Math.min(drag.startX, drag.currentX));
  const last = notchFrequencyAt(Math.max(drag.startX, drag.currentX));
  const binWidth = (state.waterfall.frequency_hz.at(-1) - state.waterfall.frequency_hz[0])
    / Math.max(1, state.waterfall.frequency_hz.length - 1);
  const notchWidth = Math.max(8, 4 * binWidth);
  const center = (first + last) / 2;
  const lowHz = last - first < 3 ? Math.max(0, center - notchWidth / 2) : first;
  const highHz = last - first < 3 ? lowHz + notchWidth : last;
  state.frequencyNotches.push({
    id: globalThis.crypto?.randomUUID?.() || `notch-${Date.now()}`,
    low_hz: lowHz,
    high_hz: highHz,
    color: NOTCH_COLORS[state.frequencyNotches.length % NOTCH_COLORS.length],
  });
  saveFrequencyNotches();
});
notchPlaceStrip.addEventListener("pointercancel", () => {
  state.notchDrag = null;
  renderFrequencyNotches();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.notchPlacement) {
    state.notchPlacement = false;
    state.notchDrag = null;
    el("notch-mode").setAttribute("aria-pressed", "false");
    el("notch-mode").textContent = "Place notch";
    renderFrequencyNotches();
  }
});

function selectTrack(track) {
  if (frequencyNotchMatches(track)) return;
  const savedRegion = matchingSavedRegionForCandidate(track);
  if (savedRegion) {
    selectSavedRegionFromCandidate(savedRegion, track);
    el("inspector-context").textContent = `Saved region · ${savedRegion.signal_id || "signal identity"} · revise its existing label or bounds.`;
    return;
  }
  saveSelectedBatchLabelDraft();
  stopSelectedAudioPlayback(null);
  state.classifierProposal = null;
  clearDecoderRecommendations();
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
  state.selectionMergeRegions = boxesCrossedBySelection(state.selection);
  updateSelectionMergeControls();
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
  setRegionSaveDisabled(state.liveView || !state.capture);
  syncLiveAnnotationSaveButton();
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
  const tracks = visibleTracks();
  el("track-count").textContent = String(tracks.length);
  list.replaceChildren();
  if (!tracks.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = state.tracks.length > 0
      ? `All ${state.tracks.length} heuristic candidate${state.tracks.length === 1 ? " is" : "s are"} suppressed by frequency notches.`
      : state.liveView && state.liveTrackModeMessage
        ? state.liveTrackModeMessage
        : "No persistent narrow-tone candidates detected. This heuristic can miss signals; manual selection is always available.";
    list.append(empty);
  } else {
    for (const track of tracks) {
      const row = document.createElement("div");
      row.className = "track-row";
      const pendingMerge = pendingWindowMergeSources().tracks.has(heuristicCandidateId(track));
      if (pendingMerge) row.classList.add("is-pending-merge");
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
      button.textContent = pendingMerge ? "Included in staged merge" : "Review region";
      button.disabled = pendingMerge;
      button.addEventListener("click", () => selectTrack(track));
      row.append(summary, detail, button);
      if (!state.liveView) {
        const adjust = document.createElement("button");
        adjust.type = "button";
        adjust.textContent = "Redraw bounds";
        adjust.title = "Draw a tighter time/frequency box on the waterfall.";
        adjust.addEventListener("click", () => armCandidateRedraw(track));
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "Remove box";
        remove.title = "Remove this provisional candidate from the current analysis.";
        remove.disabled = pendingMerge;
        remove.addEventListener("click", () => {
          state.tracks = state.tracks.filter((item) => item !== track);
          if (state.selectedTrack === track) {
            state.selectedTrack = null;
            state.selection = null;
          }
          renderTracks();
        });
        row.append(adjust, remove);
      }
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
        tryButton.disabled = pendingMerge;
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

function selectionFromCanvasCorners(x1, y1, x2, y2) {
  const data = state.waterfall;
  const view = waterfallViewBounds();
  const left = Math.max(0, Math.min(overlayCanvas.width - 2, Math.min(x1, x2)));
  const right = Math.max(left + 1, Math.min(overlayCanvas.width - 1, Math.max(x1, x2)));
  const top = Math.max(0, Math.min(overlayCanvas.height - 2, Math.min(y1, y2)));
  const bottom = Math.max(top + 1, Math.min(overlayCanvas.height - 1, Math.max(y1, y2)));
  const firstSample = sampleAtCanvasRow(top);
  const lastSample = sampleAtCanvasRow(bottom);
  const sampleStart = Math.max(0, Math.floor(Math.min(firstSample, lastSample)));
  const sampleEnd = Math.min(data.sample_count, Math.ceil(Math.max(firstSample, lastSample)) + data.fft_size);
  const frequencyIndex = (x) => Math.min(view.frequencyEnd - 1, view.frequencyStart + Math.max(0, Math.round(x)));
  const frequencyLow = data.frequency_hz[frequencyIndex(left)];
  const frequencyHigh = data.frequency_hz[frequencyIndex(Math.max(left + 1, right))];
  return {
    sample_start: sampleStart,
    sample_count: Math.max(1, sampleEnd - sampleStart),
    frequency_low_hz: frequencyLow,
    frequency_high_hz: Math.max(frequencyLow + 1, frequencyHigh),
    x1: left, y1: top, x2: right, y2: bottom,
  };
}

function resizeHandleAt(point) {
  if (!state.selection || state.liveView || state.coverageOnly || state.batchBoxMode
      || state.adjustingGeometry || (!state.editRegion && !state.batchReviewBox && !state.selectedTrack)) return null;
  const {x1, y1, x2, y2} = state.selection;
  const corners = [
    {name: "top-left", x: x1, y: y1}, {name: "top-right", x: x2, y: y1},
    {name: "bottom-left", x: x1, y: y2}, {name: "bottom-right", x: x2, y: y2},
  ];
  return corners.find((corner) => Math.hypot(point.x - corner.x, point.y - corner.y) <= 9) || null;
}

overlayCanvas.addEventListener("pointerdown", (event) => {
  if (!state.waterfall || (state.coverageOnly && !state.coverageMaskMode)) return;
  stopSelectedAudioPlayback(null);
  overlayCanvas.setPointerCapture(event.pointerId);
  const point = canvasPoint(event);
  const handle = resizeHandleAt(point);
  if (handle) {
    state.resizeDrag = {
      handle: handle.name,
      originalSelection: {...state.selection},
      originalBox: state.batchReviewBox ? {
        sample_start: state.batchReviewBox.sample_start,
        sample_count: state.batchReviewBox.sample_count,
        frequency_low_hz: state.batchReviewBox.frequency_low_hz,
        frequency_high_hz: state.batchReviewBox.frequency_high_hz,
      } : null,
    };
    state.drag = null;
    event.preventDefault();
    return;
  }
  if (!state.liveView && !state.coverageOnly && !state.batchBoxMode && !state.adjustingGeometry
      && !state.adjustingCandidate && !state.drawingSignalBox && !state.splitMode
      && (state.batchReviewBox || state.editRegion || state.selectedTrack) && state.selection) {
    const {x1, y1, x2, y2} = state.selection;
    if (point.x > Math.min(x1, x2) && point.x < Math.max(x1, x2)
        && point.y > Math.min(y1, y2) && point.y < Math.max(y1, y2)) {
      state.moveDrag = {start: point, originalSelection: {...state.selection}, moved: false};
      state.drag = null;
      event.preventDefault();
      return;
    }
  }
  if (!state.adjustingGeometry && !state.adjustingCandidate && !state.batchBoxMode
      && !state.drawingSignalBox && !state.splitMode && !state.coverageMaskMode) {
    saveSelectedBatchLabelDraft();
  }
  state.drag = { ...point, x2: point.x, y2: point.y };
  state.selection = { x1: point.x, y1: point.y, x2: point.x, y2: point.y };
  state.selectionMergeRegions = [];
  updateSelectionMergeControls();
  el("hint").classList.add("hidden");
  drawOverlays();
});

overlayCanvas.addEventListener("pointermove", (event) => {
  if (state.moveDrag) {
    const point = canvasPoint(event);
    const move = state.moveDrag;
    const dx = point.x - move.start.x;
    const dy = point.y - move.start.y;
    if (!move.moved && Math.hypot(dx, dy) < 5) return;
    move.moved = true;
    const original = move.originalSelection;
    const left = Math.min(original.x1, original.x2);
    const right = Math.max(original.x1, original.x2);
    const top = Math.min(original.y1, original.y2);
    const bottom = Math.max(original.y1, original.y2);
    const shiftX = Math.max(-left, Math.min(overlayCanvas.width - 1 - right, dx));
    const shiftY = Math.max(-top, Math.min(overlayCanvas.height - 1 - bottom, dy));
    const movedSelection = selectionFromCanvasCorners(
      original.x1 + shiftX, original.y1 + shiftY,
      original.x2 + shiftX, original.y2 + shiftY,
    );
    if (state.capture) {
      const window = annotationWindowBounds(state.capture);
      movedSelection.sample_start = Math.max(window.startSample, Math.min(
        window.endSample - movedSelection.sample_count, movedSelection.sample_start,
      ));
    }
    const view = waterfallViewBounds();
    const minHz = state.waterfall.frequency_hz[view.frequencyStart];
    const maxHz = state.waterfall.frequency_hz[view.frequencyEnd - 1];
    const widthHz = Math.min(maxHz - minHz, movedSelection.frequency_high_hz - movedSelection.frequency_low_hz);
    movedSelection.frequency_low_hz = Math.max(minHz, Math.min(maxHz - widthHz, movedSelection.frequency_low_hz));
    movedSelection.frequency_high_hz = movedSelection.frequency_low_hz + widthHz;
    state.selection = movedSelection;
    reprojectSelection();
    drawOverlays();
    return;
  }
  if (state.resizeDrag) {
    const point = canvasPoint(event);
    const {handle, originalSelection} = state.resizeDrag;
    const fixedX = handle.endsWith("left") ? originalSelection.x2 : originalSelection.x1;
    const fixedY = handle.startsWith("top") ? originalSelection.y2 : originalSelection.y1;
    const movingIsLeft = handle.endsWith("left");
    const movingIsTop = handle.startsWith("top");
    const x1 = movingIsLeft ? point.x : fixedX;
    const x2 = movingIsLeft ? fixedX : point.x;
    const y1 = movingIsTop ? point.y : fixedY;
    const y2 = movingIsTop ? fixedY : point.y;
    state.selection = selectionFromCanvasCorners(x1, y1, x2, y2);
    drawOverlays();
    return;
  }
  if (!state.drag) return;
  const point = canvasPoint(event);
  state.drag.x2 = point.x;
  state.drag.y2 = point.y;
  state.selection = { x1: state.drag.x, y1: state.drag.y, x2: point.x, y2: point.y };
  drawOverlays();
});

overlayCanvas.addEventListener("pointerup", (event) => {
  if (state.resizeDrag) {
    const box = state.batchReviewBox;
    state.resizeDrag = null;
    if (box) {
      saveSelectedBatchLabelDraft();
      const previousGeometry = {
        sample_start: box.sample_start,
        sample_count: box.sample_count,
        frequency_low_hz: box.frequency_low_hz,
        frequency_high_hz: box.frequency_high_hz,
      };
      const {sample_start, sample_count, frequency_low_hz, frequency_high_hz} = state.selection;
      Object.assign(box, {sample_start, sample_count, frequency_low_hz, frequency_high_hz});
      state.batchBoxUndo.push({type: "restore-box-geometry", box, geometry: previousGeometry});
      renderBatchBoxes();
      el("geometry-status").textContent = "Draft bounds resized. Save annotations to apply the change with the rest of this window.";
    } else if (state.editRegion) {
      saveSelectedExistingRegionDraft();
      state.geometryMergeRegions = boxesCrossedBySelection(state.selection, state.editRegion.id);
      state.selectionMergeRegions = [...state.geometryMergeRegions];
      el("geometry-status").textContent = "Bounds resized and staged in this window. Save annotations to apply the change.";
      const crossed = state.geometryMergeRegions.length;
      el("merge-redrawn-regions").hidden = crossed < 1 || crossed > 31;
      el("merge-redrawn-regions").textContent = `Merge this signal and ${crossed} crossed box${crossed === 1 ? "" : "es"}`;
      el("geometry-merge-status").hidden = crossed < 1;
      el("geometry-merge-status").textContent = crossed
        ? `Resize crosses ${crossed} saved or CV box${crossed === 1 ? "" : "es"}. You can merge them into the resized signal before saving.`
        : "";
    } else if (state.selectedTrack) {
      const geometry = {
        sample_start: state.selection.sample_start,
        sample_count: state.selection.sample_count,
        frequency_low_hz: state.selection.frequency_low_hz,
        frequency_high_hz: state.selection.frequency_high_hz,
      };
      Object.assign(state.selectedTrack, geometry, {
        time_start_seconds: geometry.sample_start / state.waterfall.sample_rate,
        time_end_seconds: (geometry.sample_start + geometry.sample_count) / state.waterfall.sample_rate,
      });
      const proposalId = state.selectedTrack.cv_proposal_id;
      const detection = proposalId && state.cvDetections.find((item) => (item.cv_proposal_id || item.proposal_id) === proposalId);
      if (detection) {
        detection._originalProposalBounds ||= {
          sample_start: detection.sample_start, sample_count: detection.sample_count,
          frequency_low_hz: detection.frequency_low_hz, frequency_high_hz: detection.frequency_high_hz,
        };
        Object.assign(detection, geometry);
        renderCvDetections();
      }
      el("selection-details").textContent = `Candidate bounds resized · ${(geometry.sample_start / state.waterfall.sample_rate).toFixed(2)}–${((geometry.sample_start + geometry.sample_count) / state.waterfall.sample_rate).toFixed(2)} s · save as a signal to keep this correction.`;
      state.selectionMergeRegions = boxesCrossedBySelection(state.selection);
    }
    updateAnnotationSaveAvailability();
    updateSelectionMergeControls();
    drawOverlays();
    return;
  }
  if (state.moveDrag) {
    const move = state.moveDrag;
    state.moveDrag = null;
    if (!move.moved) {
      state.selection = move.originalSelection;
      const sample = sampleAtCanvasRow(move.start.y);
      const view = waterfallViewBounds();
      const frequency = state.waterfall.frequency_hz[Math.min(view.frequencyEnd - 1, view.frequencyStart + move.start.x)];
      const draftIndex = state.batchBoxes.findLastIndex((box) => sample >= box.sample_start
        && sample <= box.sample_start + box.sample_count
        && frequency >= box.frequency_low_hz && frequency <= box.frequency_high_hz);
      if (draftIndex >= 0 && !state.batchBoxMode) reviewBatchBox(draftIndex);
      else {
        const saved = state.regions.find((region) => region.lineage_state !== "superseded"
          && !state.windowRegionDeletes.has(region.id)
          && sample >= region.sample_start && sample <= region.sample_start + region.sample_count
          && frequency >= region.frequency_low_hz && frequency <= region.frequency_high_hz);
        if (saved) selectSavedRegion(saved);
        else {
          const track = hitSignal(visibleTracks(), sample, frequency);
          if (track) selectTrack(track);
        }
      }
      return;
    } else {
      const box = state.batchReviewBox;
      const geometry = {
        sample_start: state.selection.sample_start,
        sample_count: state.selection.sample_count,
        frequency_low_hz: state.selection.frequency_low_hz,
        frequency_high_hz: state.selection.frequency_high_hz,
      };
      if (box) {
        saveSelectedBatchLabelDraft();
        const previousGeometry = {
          sample_start: box.sample_start, sample_count: box.sample_count,
          frequency_low_hz: box.frequency_low_hz, frequency_high_hz: box.frequency_high_hz,
        };
        Object.assign(box, geometry);
        state.batchBoxUndo.push({type: "restore-box-geometry", box, geometry: previousGeometry});
        renderBatchBoxes();
        el("geometry-status").textContent = "Draft moved. Save annotations to apply the change with the rest of this window.";
      } else if (state.editRegion) {
        saveSelectedExistingRegionDraft();
        state.geometryMergeRegions = boxesCrossedBySelection(state.selection, state.editRegion.id);
        state.selectionMergeRegions = [...state.geometryMergeRegions];
        const crossed = state.geometryMergeRegions.length;
        el("geometry-status").textContent = "Signal moved. Save annotations to apply the change with the rest of this window.";
        el("merge-redrawn-regions").hidden = crossed < 1 || crossed > 31;
        el("merge-redrawn-regions").textContent = `Merge this signal and ${crossed} crossed box${crossed === 1 ? "" : "es"}`;
        el("geometry-merge-status").hidden = crossed < 1;
        el("geometry-merge-status").textContent = crossed
          ? `Move crosses ${crossed} saved or CV box${crossed === 1 ? "" : "es"}. You can merge them into the selected signal before saving.`
          : "";
      } else if (state.selectedTrack) {
        Object.assign(state.selectedTrack, geometry, {
          time_start_seconds: geometry.sample_start / state.waterfall.sample_rate,
          time_end_seconds: (geometry.sample_start + geometry.sample_count) / state.waterfall.sample_rate,
        });
        const proposalId = state.selectedTrack.cv_proposal_id;
        const detection = proposalId && state.cvDetections.find((item) => (item.cv_proposal_id || item.proposal_id) === proposalId);
        if (detection) {
          detection._originalProposalBounds ||= {
            sample_start: detection.sample_start, sample_count: detection.sample_count,
            frequency_low_hz: detection.frequency_low_hz, frequency_high_hz: detection.frequency_high_hz,
          };
          Object.assign(detection, geometry);
          renderCvDetections();
        }
        state.selectionMergeRegions = boxesCrossedBySelection(state.selection);
        el("selection-details").textContent = "Candidate moved · save as a signal to keep the corrected bounds.";
      }
      updateAnnotationSaveAvailability();
      updateSelectionMergeControls();
      drawOverlays();
      return;
    }
  }
  if (!state.drag) return;
  const point = canvasPoint(event);
  const data = state.waterfall;
  const view = waterfallViewBounds();
  const rect = overlayCanvas.getBoundingClientRect();
  const moved = Math.hypot((state.drag.x - point.x) * rect.width / overlayCanvas.width, (state.drag.y - point.y) * rect.height / overlayCanvas.height);
  if (moved < 5 && !state.splitMode && !state.coverageMaskMode
      && !state.adjustingCandidate && !state.drawingSignalBox) {
    const sample = sampleAtCanvasRow(point.y);
    const frequency = data.frequency_hz[Math.min(view.frequencyEnd - 1, view.frequencyStart + point.x)];
    const draftIndex = state.batchBoxes.findLastIndex((box) => sample >= box.sample_start
      && sample <= box.sample_start + box.sample_count
      && frequency >= box.frequency_low_hz && frequency <= box.frequency_high_hz);
    if (draftIndex >= 0 && !state.batchBoxMode) {
      state.drag = null;
      reviewBatchBox(draftIndex);
      return;
    }
    const track = hitSignal(visibleTracks(), sample, frequency);
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
  if (state.adjustingCandidate) {
    const candidate = state.adjustingCandidate;
    Object.assign(candidate, {
      sample_start: sampleStart,
      sample_count: Math.max(1, sampleEnd - sampleStart),
      frequency_low_hz: frequencyLow,
      frequency_high_hz: frequencyHigh,
      time_start_seconds: sampleStart / data.sample_rate,
      time_end_seconds: sampleEnd / data.sample_rate,
    });
    state.adjustingCandidate = null;
    state.drawingSignalBox = false;
    updateSingleBoxButton();
    state.drag = null;
    if (state.cvDetections.includes(candidate)) renderCvDetections();
    else renderTracks();
    if (state.cvDetections.includes(candidate)) selectCvDetection(candidate);
    else selectTrack(candidate);
    el("selection-details").textContent = `Adjusted provisional candidate bounds · ${(sampleStart / data.sample_rate).toFixed(2)}–${(sampleEnd / data.sample_rate).toFixed(2)} s · ${frequencyLow.toFixed(0)}–${frequencyHigh.toFixed(0)} Hz. Save as a region to keep this correction.`;
    drawOverlays();
    return;
  }
  const wasDrawingSignalBox = state.drawingSignalBox;
  state.drawingSignalBox = false;
  updateSingleBoxButton();
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
  if (state.batchBoxMode) {
    const { x1, y1, x2, y2, ...box } = state.selection;
    const draftBox = {...box, client_id: crypto.randomUUID()};
    state.batchBoxUndo.push({type: "add", box: draftBox});
    state.batchBoxes.push(draftBox);
    state.selection = null;
    state.drag = null;
    state.classifierProposal = null;
    renderBatchBoxes();
    setRegionSaveDisabled(true);
    updateModelButton();
    updateSceneReviewAvailability();
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
    state.geometryMergeRegions = boxesCrossedBySelection(state.selection, state.editRegion.id);
    state.selectionMergeRegions = [...state.geometryMergeRegions];
    el("save-region-bounds").hidden = false;
    el("cancel-region-bounds").hidden = false;
    el("geometry-status").textContent = `Proposed bounds · samples ${sampleStart.toLocaleString()}–${sampleEnd.toLocaleString()} · ${frequencyLow.toFixed(0)}–${frequencyHigh.toFixed(0)} Hz. Save to append a geometry revision.`;
    const mergeCount = state.geometryMergeRegions.length;
    const unknownBandCount = state.geometryMergeRegions.filter((source) => {
      const region = source.kind === "region" ? source.region : source.detection;
      return !Number.isFinite(region.frequency_low_hz) || !Number.isFinite(region.frequency_high_hz);
    }).length;
    const mergeButton = el("merge-redrawn-regions");
    mergeButton.hidden = mergeCount < 1 || mergeCount > 31;
    mergeButton.textContent = `Merge this signal and ${mergeCount} crossed box${mergeCount === 1 ? "" : "es"}`;
    const mergeStatus = el("geometry-merge-status");
    mergeStatus.hidden = mergeCount < 1;
    mergeStatus.textContent = mergeCount > 31
      ? `This redraw crosses ${mergeCount} other saved or CV boxes. Merge includes this selected signal and up to 31 crossed boxes.`
      : mergeCount > 0
        ? `This redraw crosses ${mergeCount} other saved or CV boxes. Merge keeps your drawn bounds and includes the selected signal${unknownBandCount ? `; ${unknownBandCount} crossed boxes have unknown frequency bounds and are matched by time overlap` : ""}.`
        : "";
    el("selection-summary").textContent = "Proposed corrected bounds";
    el("selection-details").textContent = "Original geometry remains in history until these bounds are saved.";
    drawOverlays();
    return;
  }
  state.selectedTrack = null;
  beginNewRegion();
  el("batch-unknown-row").hidden = false;
  state.selectionMergeRegions = boxesCrossedBySelection(state.selection);
  updateSelectionMergeControls();
  openInspector("label");
  el("selection-summary").textContent = `${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz`;
  const t0 = sampleStart / data.sample_rate;
  const t1 = sampleEnd / data.sample_rate;
  el("selection-details").textContent = `Samples ${sampleStart.toLocaleString()}–${sampleEnd.toLocaleString()} · ${t0.toFixed(3)}–${t1.toFixed(3)} s`;
  if (wasDrawingSignalBox) {
    el("box-draw-status").textContent = `Box selected · ${t0.toFixed(2)}–${t1.toFixed(2)} s · ${frequencyLow.toFixed(0)}–${frequencyHigh.toFixed(0)} Hz. Save region to keep it.`;
  }
  el("inspector-context").textContent = `Manual region · ${frequencyLow.toFixed(0)}–${frequencyHigh.toFixed(0)} Hz · ${t0.toFixed(2)}–${t1.toFixed(2)} s${state.liveView ? " · stop recording to save labels" : ""}`;
  setRegionSaveDisabled(state.liveView || !state.capture);
  syncLiveAnnotationSaveButton();
  updateModelButton();
  drawOverlays();
});

overlayCanvas.addEventListener("pointercancel", () => {
  if (state.moveDrag) {
    state.selection = {...state.moveDrag.originalSelection};
    state.moveDrag = null;
    drawOverlays();
  }
  if (state.resizeDrag) {
    state.selection = {...state.resizeDrag.originalSelection};
    state.resizeDrag = null;
    drawOverlays();
  }
  state.drag = null;
});

el("save-annotations-close").addEventListener("click", () => {
  if (el("save-annotations-close").disabled) return;
  saveSelectedBatchLabelDraft();
  if (state.selection && !state.editRegion && !state.batchReviewBox && !state.batchBoxMode) stageCurrentSelectionAsWindowDraft();
  if (state.liveView) { el("save-annotations").click(); return; }
  saveWindowAnnotationDrafts().then(() => {
    el("inspector").classList.add("hidden");
  }).catch((error) => {
    showError(error.message);
    el("batch-box-status").textContent = `Could not save annotations · ${error.message}. Your drafts are still here.`;
  });
});

el("save-annotations").addEventListener("click", () => {
  if (el("save-annotations").disabled) return;
  saveSelectedBatchLabelDraft();
  if (!state.liveView) stageCurrentSelectionAsWindowDraft();
  if (!state.liveView && (state.batchBoxes.length || state.windowRegionEdits.size || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length)) {
    saveWindowAnnotationDrafts().catch((error) => {
      showError(error.message);
      el("batch-box-status").textContent = `Could not save annotations · ${error.message}. Your drafts are still here.`;
      el("label-state").textContent = `Save failed · ${error.message}. Your drafts are still here.`;
    });
    return;
  }
  if (state.liveView) {
    const selection = state.selection;
    const requestBody = {
      session_id: state.liveSessionId,
      sample_start: selection.sample_start,
      sample_count: selection.sample_count,
      frequency_low_hz: selection.frequency_low_hz,
      frequency_high_hz: selection.frequency_high_hz,
      mode_id: state.labelModeId || null,
      family: el("family").value.trim() || null,
      method: el("method").value.trim() || null,
      protocol: el("protocol").value.trim() || null,
      variant: el("variant").value.trim() || null,
      notes: el("notes").value.trim(),
    };
    el("save-annotations").disabled = true;
    el("label-state").textContent = "Saving the current 10-second live window as a training candidate…";
    request("/api/live/training-examples", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(requestBody),
    }).then((result) => {
      el("label-state").textContent = result.message;
      el("box-draw-status").textContent = `Training candidate saved · ${result.capture_id.slice(0, 8)} · review its 10-second window before export.`;
      state.selection = null;
      beginNewRegion();
      el("selection-summary").textContent = "No region selected";
      el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
      drawOverlays();
      syncLiveAnnotationSaveButton();
    }).catch((error) => {
      showError(error.message);
      el("label-state").textContent = `Could not save live training candidate · ${error.message}`;
      syncLiveAnnotationSaveButton();
    });
    return;
  }
  state.closeInspectorAfterSave = false;
  state.returnInspectorTabAfterSave = "label";
  el("save-region").click();
});

el("save-annotations-complete").addEventListener("click", async () => {
  if (el("save-annotations-complete").disabled || !state.capture?.source_metadata?.auto_review_queue) return;
  saveSelectedBatchLabelDraft();
  const hasUncommittedAnnotations = state.batchBoxes.length || state.windowRegionEdits.size
    || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length
    || (state.selection && !state.editRegion && !state.batchReviewBox && !state.batchBoxMode);
  if (state.pendingCaptureCompletion === state.capture.id && !hasUncommittedAnnotations) {
    const capture = state.capture;
    const button = el("save-annotations-complete");
    button.disabled = true;
    el("label-state").textContent = "Recording complete 10-second window coverage…";
    try {
      await recordCompleteReviewWindowCoverage(capture);
      if (state.capture?.id !== capture.id) return;
      state.pendingCaptureCompletion = null;
      el("inspector").classList.add("hidden");
      await refreshLiveReviewQueue();
      el("review-queue-dialog").showModal();
    } catch (error) {
      el("label-state").textContent = `Annotations are saved, but review completion failed · ${error.message}`;
    } finally {
      button.disabled = state.liveView || !state.capture;
    }
    return;
  }
  if (state.pendingCaptureCompletion === state.capture.id) {
    state.pendingCaptureCompletion = null;
    el("save-annotations-complete").textContent = "Save and finish";
  }
  stageCurrentSelectionAsWindowDraft();
  const completionWindow = annotationWindowBounds(state.capture);
  const unresolvedSavedRegion = state.regions.find((region) => {
    if (region.lineage_state === "superseded") return false;
    if (pendingWindowMergeSources().regions.has(region.id)) return false;
    if (state.windowRegionDeletes.has(region.id)) return false;
    if (region.sample_start < completionWindow.startSample
      || region.sample_start + region.sample_count > completionWindow.endSample) return false;
    const draft = state.windowRegionEdits.get(region.id)?.labelDraft;
    const hasIdentity = Boolean(draft
      ? draft.mode_id || draft.family || draft.method || draft.protocol || draft.variant
      : region.mode_id || region.family || region.method || region.protocol || region.variant);
    const identityUnknown = draft?.identity_unknown_confirmed ?? Boolean(region.evidence?.identity_unknown_confirmed);
    return !hasIdentity && !identityUnknown;
  });
  if (unresolvedSavedRegion) {
    if (state.editRegion?.id !== unresolvedSavedRegion.id) selectSavedRegion(unresolvedSavedRegion);
    el("label-state").textContent = "This saved signal still needs a label. Choose a mode or mark its identity unknown, then save the window changes before completion.";
    return;
  }
  if (state.batchBoxes.length || state.windowRegionEdits.size || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length || !state.selection) {
    const capture = state.capture;
    const unlabeled = state.batchBoxes.findIndex((box) => {
      const label = box === state.batchReviewBox ? readBatchLabelDraft() : box.labelDraft || {};
      return !label.identity_unknown_confirmed && ![label.mode_id, label.family, label.method, label.protocol, label.variant].some(Boolean);
    });
    if (unlabeled >= 0) {
      el("label-state").textContent = "Label every draft signal or explicitly mark its identity unknown before completing this window.";
      reviewBatchBox(unlabeled);
      return;
    }
    const button = el("save-annotations-complete");
    button.disabled = true;
    try {
      if (state.batchBoxes.length || state.windowRegionEdits.size || state.windowRegionDeletes.size || state.windowMerges.length || state.windowIdentityLinks.length) await saveWindowAnnotationDrafts();
      await recordCompleteReviewWindowCoverage(capture);
      if (state.capture?.id !== capture.id) return;
      el("inspector").classList.add("hidden");
      await refreshLiveReviewQueue();
      el("review-queue-dialog").showModal();
    } catch (error) {
      showError(error.message);
      el("label-state").textContent = `Could not complete this window · ${error.message}. Unsaved drafts remain here.`;
    } finally {
      updateAnnotationSaveAvailability();
    }
    return;
  }
  state.closeInspectorAfterSave = true;
  state.markCaptureCompleteAfterSave = true;
  el("save-region").click();
});

el("save-region").addEventListener("click", async () => {
  if (!state.selection || !state.capture) return;
  const closeInspectorAfterSave = Boolean(state.closeInspectorAfterSave);
  const markCaptureCompleteAfterSave = Boolean(state.markCaptureCompleteAfterSave);
  state.closeInspectorAfterSave = false;
  state.markCaptureCompleteAfterSave = false;
  if (!state.liveView && (state.editRegion || state.batchReviewBox)) {
    const capture = state.capture;
    saveSelectedBatchLabelDraft();
    setRegionSaveDisabled(true);
    try {
      await saveWindowAnnotationDrafts();
      if (markCaptureCompleteAfterSave) {
        await recordCompleteReviewWindowCoverage(capture);
        state.pendingCaptureCompletion = null;
      }
      if (state.capture?.id !== capture.id) return;
      if (closeInspectorAfterSave) {
        el("inspector").classList.add("hidden");
        if (markCaptureCompleteAfterSave) {
          await refreshLiveReviewQueue();
          el("review-queue-dialog").showModal();
        }
      }
      return;
    } catch (error) {
      if (markCaptureCompleteAfterSave && !state.batchBoxes.length
        && !state.windowRegionEdits.size && !state.windowRegionDeletes.size && !state.windowMerges.length && !state.windowIdentityLinks.length) {
        state.pendingCaptureCompletion = capture.id;
        el("save-annotations-complete").textContent = "Retry finish";
        el("label-state").textContent = `Annotations are saved, but window completion failed · ${error.message}`;
      } else {
        showError(error.message);
        el("label-state").textContent = `Could not save the window changes · ${error.message}. Your draft remains here.`;
      }
      setRegionSaveDisabled(false);
      return;
    }
  }
  clearError();
  const capture = state.capture;
  const captureId = capture.id;
  let annotationSaved = false;
  const cvProposalId = !state.editRegion ? state.selectedTrack?.cv_proposal_id : null;
  const edit = state.editRegion;
  const completedBatchBox = edit ? null : state.batchReviewBox;
  setRegionSaveDisabled(true);
  try {
    const selection = state.selection;
    let updatedExisting = false;
    let duplicateToUpdate = null;
    if (!edit) {
      const overlapArea = (region) => {
        const t0 = Math.max(selection.sample_start, region.sample_start);
        const t1 = Math.min(selection.sample_start + selection.sample_count, region.sample_start + region.sample_count);
        const f0 = Math.max(selection.frequency_low_hz, region.frequency_low_hz);
        const f1 = Math.min(selection.frequency_high_hz, region.frequency_high_hz);
        return Math.max(0, t1 - t0) * Math.max(0, f1 - f0);
      };
      const selectedArea = selection.sample_count * (selection.frequency_high_hz - selection.frequency_low_hz);
      const pendingMergeRegions = pendingWindowMergeSources().regions;
      const duplicate = state.regions.find((region) => {
        if (region.lineage_related || region.lineage_state === "superseded"
          || state.windowRegionDeletes.has(region.id) || pendingMergeRegions.has(region.id)) return false;
        const geometry = state.windowRegionEdits.get(region.id)?.geometry || region;
        const regionArea = geometry.sample_count * (geometry.frequency_high_hz - geometry.frequency_low_hz);
        const smallerArea = Math.min(selectedArea, regionArea);
        return smallerArea > 0 && overlapArea(geometry) / smallerArea >= 0.65;
      });
      if (duplicate && window.confirm(`This box substantially overlaps the saved “${duplicate.method || duplicate.family || "Unknown signal"}” region. Update its bounds instead of creating a duplicate? Choose Cancel to create a separate region.`)) {
        await request(`/api/captures/${captureId}/regions/${duplicate.id}/geometry`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expected_revision: duplicate.geometry_revision, sample_start: selection.sample_start, sample_count: selection.sample_count, frequency_low_hz: selection.frequency_low_hz, frequency_high_hz: selection.frequency_high_hz }),
        });
        updatedExisting = true;
        duplicateToUpdate = duplicate;
      }
    }
    const classifierProposal = state.classifierProposal?.capture_id === captureId
      && state.classifierProposal.selection === selection
      ? state.classifierProposal.result : null;
    const taxonomyDraft = Boolean(state.labelModeId
      || ["family", "method", "protocol", "variant"].some((field) => el(field).value.trim()));
    const notesDraft = el("notes").value.trim();
    const hasAnnotationDraft = taxonomyDraft || Boolean(notesDraft) || Boolean(classifierProposal);
    const destination = edit
      ? `/api/captures/${captureId}/regions/${edit.id}/labels`
      : updatedExisting
        ? `/api/captures/${captureId}/regions/${duplicateToUpdate.id}/labels`
        : `/api/captures/${captureId}/regions`;
    if (!updatedExisting || hasAnnotationDraft) await request(destination, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...(edit || updatedExisting ? {expected_revision: edit?.revision ?? duplicateToUpdate.revision} : {}),
        sample_start: selection.sample_start,
        sample_count: selection.sample_count,
        frequency_low_hz: selection.frequency_low_hz,
        frequency_high_hz: selection.frequency_high_hz,
        mode_id: state.labelModeId,
        origin: !edit && state.selectedTrack ? "detector_proposal" : el("missed-signal").checked && !edit ? "user_added_missed_detection" : "manual",
        detector_evidence: !edit ? state.selectedTrack?.detector_evidence || null : null,
        signal_presence_confirmed: !edit && Boolean(state.selectedTrack?.detector_evidence) && el("cv-presence-confirmed").checked,
        identity_unknown_confirmed: el("batch-unknown-confirmed").checked,
        classifier_proposal: classifierProposal ? {
          sample_start: classifierProposal.sample_start,
          sample_count: classifierProposal.sample_count,
          selection_sample_count: classifierProposal.selection_sample_count,
          frequency_low_hz: classifierProposal.selection_frequency_range_hz?.[0],
          frequency_high_hz: classifierProposal.selection_frequency_range_hz?.[1],
          crop_count: classifierProposal.requested_crop_count || 1,
        } : null,
        family: el("family").value.trim() || (updatedExisting && !taxonomyDraft ? duplicateToUpdate.family : null),
        method: el("method").value.trim() || (updatedExisting && !taxonomyDraft ? duplicateToUpdate.method : null),
        protocol: el("protocol").value.trim() || (updatedExisting && !taxonomyDraft ? duplicateToUpdate.protocol : null),
        variant: el("variant").value.trim() || (updatedExisting && !taxonomyDraft ? duplicateToUpdate.variant : null),
        notes: notesDraft || (updatedExisting ? duplicateToUpdate.notes : ""),
      }),
    });
    annotationSaved = true;
    if (markCaptureCompleteAfterSave) {
      await recordCompleteReviewWindowCoverage(capture);
      state.pendingCaptureCompletion = null;
    }
    const regions = await request(`/api/captures/${captureId}/regions`);
    if (state.capture?.id !== captureId) return;
    state.regions = regions;
    if (cvProposalId && state.regions.some((region) => region.evidence?.detector_proposal?.proposal_id === cvProposalId)) {
      removeSavedCvProposal(cvProposalId);
      el("cv-status").textContent = "Saved signal added · its CV proposal is now attached as detector evidence.";
    }
    if (completedBatchBox) {
      state.batchBoxes = state.batchBoxes.filter((box) => box !== completedBatchBox);
      state.batchBoxUndo = [];
      renderBatchBoxes();
    }
    state.classifierProposal = null;
    clearDecoderRecommendations();
    state.selection = null;
    beginNewRegion();
    renderRegions();
    await refreshSceneCoverage(captureId);
    el("coverage-panel").hidden = true;
    updateModelButton();
    el("selection-summary").textContent = "No region selected";
    el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
    for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = "";
    el("missed-signal").checked = false;
    drawOverlays();
    if (closeInspectorAfterSave) {
      el("inspector").classList.add("hidden");
      if (markCaptureCompleteAfterSave) {
        await refreshLiveReviewQueue();
        el("review-queue-dialog").showModal();
      }
    } else if (completedBatchBox && state.batchBoxes.length) {
      reviewBatchBox(0);
    } else {
      if (completedBatchBox) {
        el("batch-box-status").textContent = "All drawn selections are saved. Review the saved signals above to accept or correct their labels.";
      }
      el("box-draw-status").textContent = updatedExisting
        ? "Updated the existing signal bounds. The saved signals list is current."
        : edit
          ? "Label revision saved. The saved signals list is current."
          : "Region saved. Review it in Saved signals above.";
      const returnTab = state.returnInspectorTabAfterSave || "signals";
      state.returnInspectorTabAfterSave = null;
      openInspector(returnTab);
    }
  } catch (error) {
    if (state.capture?.id !== captureId) return;
    showError(error.message);
    if (markCaptureCompleteAfterSave && annotationSaved) {
      state.pendingCaptureCompletion = captureId;
      el("save-annotations-complete").textContent = "Retry finish";
      el("label-state").textContent = "Annotations were saved, but window completion failed. Retry to record complete 10-second coverage.";
    }
    if (error.status === 409) {
      state.regions = await request(`/api/captures/${captureId}/regions`);
      renderRegions();
      await refreshSceneCoverage(captureId);
      el("label-state").textContent = "A newer label exists. Reopen its marker to review it; your draft is retained here.";
    }
    state.closeInspectorAfterSave = false;
    setRegionSaveDisabled(false);
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
  state.geometryMergeRegions = [];
  el("merge-redrawn-regions").hidden = true;
  el("geometry-merge-status").hidden = true;
  el("geometry-merge-status").textContent = "";
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
    saveSelectedExistingRegionDraft();
    state.adjustingGeometry = false;
    state.geometryDraft = false;
    button.hidden = true;
    el("cancel-region-bounds").hidden = true;
    el("geometry-status").textContent = "Bounds staged in this window draft. Save annotations to apply them with the other changes.";
    updateAnnotationSaveAvailability();
    renderRegions();
    drawOverlays();
  } catch (error) {
    el("geometry-status").textContent = `Could not stage corrected bounds: ${error.message}`;
    button.hidden = false;
    el("cancel-region-bounds").hidden = false;
  } finally {
    button.disabled = false;
  }
});

el("merge-redrawn-regions").addEventListener("click", async () => {
  const pending = pendingWindowMergeSources();
  const crossed = state.geometryMergeRegions.filter((source) => source.kind === "cv"
    ? state.cvDetections.includes(source.detection)
      && !pending.proposals.has(source.detection.cv_proposal_id || source.detection.proposal_id)
      && !state.excludedMergeCandidates.has(mergeCandidateKey(source))
    : source.kind === "draft"
      ? state.batchBoxes.includes(source.box) && !state.excludedMergeCandidates.has(mergeCandidateKey(source))
      : state.regions.some((current) => current.id === source.region.id
      && current.geometry_revision === source.region.geometry_revision
      && current.lineage_state !== "superseded")
      && !pending.regions.has(source.region.id)
      && !state.excludedMergeCandidates.has(mergeCandidateKey(source)));
  const selected = state.regions.find((region) => region.id === state.editRegion?.id && region.lineage_state === "active");
  const sources = [...(selected ? [{kind: "region", region: selected}] : []), ...crossed];
  const selection = state.selection;
  if (sources.length < 2 || sources.length > 32 || !selection) return;
  await mergeRegionGroup(sources, {
    sample_start: selection.sample_start,
    sample_count: selection.sample_count,
    frequency_low_hz: selection.frequency_low_hz,
    frequency_high_hz: selection.frequency_high_hz,
  }, el("merge-redrawn-regions"));
});

el("cancel-region-bounds").addEventListener("click", () => {
  if (!state.capture || !state.editRegion) return;
  const current = state.regions.find((region) => region.id === state.editRegion.id);
  if (current) {
    selectSavedRegion(current);
    openInspector("signals");
  }
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

async function mergeRegionGroup(sources, resultBounds = null, triggerButton = el("merge-regions")) {
  const captureId = state.capture?.id;
  if (!captureId || sources.length > 32) return;
  saveSelectedBatchLabelDraft();
  const regions = sources.filter((source) => source.kind === "region" || (!source.kind && source.id)).map((source) => source.region || source);
  let proposals = sources.filter((source) => source.kind === "cv").map(({ detection }) => {
    const bounds = detection._originalProposalBounds || detection;
    return {
      proposal_id: detection.cv_proposal_id || detection.proposal_id,
      decision: "dismissed",
      sample_start: bounds.sample_start,
      sample_count: bounds.sample_count,
      frequency_low_hz: bounds.frequency_low_hz,
      frequency_high_hz: bounds.frequency_high_hz,
      confidence: detection.confidence,
      detector_evidence: detection.detector_evidence || detectorEvidenceFor(detection),
      note: "Merged into the user-drawn signal selection; review feedback only, not a negative label.",
    };
  });
  let heuristicProposals = sources.filter((source) => source.kind === "track").map(({track}) => ({
    candidate_id: heuristicCandidateId(track),
    sample_start: track.sample_start,
    sample_count: track.sample_count,
    frequency_low_hz: track.frequency_low_hz,
    frequency_high_hz: track.frequency_high_hz,
    modulation_hypothesis: track.modulation_hypothesis || "unknown",
    tone_count_candidate: track.tone_count_candidate || 1,
    tone_centers_hz: track.tone_centers_hz || [],
    peak_snr_db: Number.isFinite(track.peak_snr_db) ? track.peak_snr_db : null,
  }));
  if (!sources.length && !resultBounds) return;
  const capture = state.capture;
  const window = annotationWindowBounds(capture);
  const explicitDraftSources = sources.filter((source) => source.kind === "draft").map((source) => source.box);
  const sourceBounds = [...regions, ...proposals, ...heuristicProposals, ...explicitDraftSources];
  const bounds = resultBounds || {
    sample_start: Math.min(...sourceBounds.map((item) => item.sample_start)),
    sample_count: Math.max(...sourceBounds.map((item) => item.sample_start + item.sample_count))
      - Math.min(...sourceBounds.map((item) => item.sample_start)),
    frequency_low_hz: Math.min(...sourceBounds.map((item) => item.frequency_low_hz)),
    frequency_high_hz: Math.max(...sourceBounds.map((item) => item.frequency_high_hz)),
  };
  if (bounds.sample_start < window.startSample || bounds.sample_start + bounds.sample_count > window.startSample + window.sampleCount) {
    el("merge-status").textContent = "The merged signal must fit inside the current 10-second window.";
    return;
  }
  const timeEnd = bounds.sample_start + bounds.sample_count;
  const intersects = (box) => {
    const boxEnd = box.sample_start + box.sample_count;
    return box.sample_start <= timeEnd && boxEnd >= bounds.sample_start
      && box.frequency_low_hz <= bounds.frequency_high_hz && box.frequency_high_hz >= bounds.frequency_low_hz;
  };
  const previousBoxes = [...state.batchBoxes];
  const previousMerges = [...state.windowMerges];
  const explicitlySelectedDrafts = explicitDraftSources;
  const affectedDrafts = resultBounds
    ? state.batchBoxes.filter((box) => intersects(box)
      && (box === state.batchReviewBox || !state.excludedMergeCandidates.has(`draft:${box.draftMergeId || box.client_id}`)))
    : explicitlySelectedDrafts;
  const selectedDraft = affectedDrafts.includes(state.batchReviewBox) ? state.batchReviewBox : null;
  for (const box of affectedDrafts) {
    if (!box.mergeClientId) continue;
    const priorMerge = state.windowMerges.find((merge) => merge.client_id === box.mergeClientId);
    if (priorMerge) {
      regions.push(...priorMerge.source_region_ids.map((id) => state.regions.find((region) => region.id === id)).filter(Boolean));
      proposals.push(...priorMerge.cv_proposals);
      heuristicProposals.push(...(priorMerge.heuristic_proposals || []));
    }
  }
  const uniqueRegions = [...new Map(regions.map((region) => [region.id, region])).values()];
  proposals = [...new Map(proposals.map((proposal) => [proposal.proposal_id, proposal])).values()];
  heuristicProposals = [...new Map(heuristicProposals.map((proposal) => [proposal.candidate_id, proposal])).values()];
  if (uniqueRegions.length + proposals.length + heuristicProposals.length < 2 && !resultBounds && affectedDrafts.length < 2) return;
  if (uniqueRegions.length + proposals.length + heuristicProposals.length > 32) {
    el("merge-status").textContent = "A merge can contain up to 32 saved or provisional boxes at a time.";
    return;
  }
  const selectedRegionId = state.editRegion?.id || null;
  const targetRegionId = uniqueRegions.some((region) => region.id === selectedRegionId) ? selectedRegionId : null;
  const labelDraft = selectedDraft?.labelDraft ? {...selectedDraft.labelDraft} : resultBounds ? readBatchLabelDraft() : {
    mode_id: null, mode_name: "", family: "", method: "", protocol: "", variant: "", notes: "",
    identity_unknown_confirmed: false,
  };
  if (targetRegionId && !labelDraft.mode_id && !labelDraft.family && !labelDraft.method && !labelDraft.protocol && !labelDraft.variant && !labelDraft.identity_unknown_confirmed) {
    const target = regions.find((region) => region.id === targetRegionId);
    if (target) Object.assign(labelDraft, {
      mode_id: target.mode_id || null, mode_name: target.mode_id || "",
      family: target.family || "", method: target.method || "", protocol: target.protocol || "",
      variant: target.variant || "", notes: target.notes || "",
      identity_unknown_confirmed: Boolean(target.evidence?.identity_unknown_confirmed),
    });
  }
  const clientId = crypto.randomUUID();
  const priorSourceEdits = uniqueRegions.map((region) => state.windowRegionEdits.get(region.id)).filter(Boolean);
  for (const region of uniqueRegions) state.windowRegionEdits.delete(region.id);
  const mergeDraft = uniqueRegions.length + proposals.length + heuristicProposals.length ? {
    client_id: clientId,
    source_region_ids: uniqueRegions.map((region) => region.id),
    expected_revisions: Object.fromEntries(uniqueRegions.map((region) => [region.id, region.revision])),
    expected_geometry_revisions: Object.fromEntries(uniqueRegions.map((region) => [region.id, region.geometry_revision])),
    cv_proposals: proposals,
    heuristic_proposals: heuristicProposals,
    target_region_id: targetRegionId,
    ...bounds,
    labelDraft,
  } : null;
  const resultBox = {
    client_id: `merge-${clientId}`,
    draftMergeId: clientId,
    mergeClientId: mergeDraft ? clientId : null,
    ...bounds,
    labelDraft,
    sourceCount: uniqueRegions.length + proposals.length + heuristicProposals.length + affectedDrafts.length,
  };
  const previousSelection = state.batchReviewBox;
  const previousEditRegion = state.editRegion;
  const previousGeometrySelection = state.selection ? {...state.selection} : null;
  if (mergeDraft) state.windowMerges.push(mergeDraft);
  state.windowMerges = state.windowMerges.filter((merge) => !affectedDrafts.some((box) => box.mergeClientId === merge.client_id));
  state.batchBoxes = state.batchBoxes.filter((box) => !affectedDrafts.includes(box));
  state.batchBoxes.push(resultBox);
  state.batchBoxUndo.push({type: "undo-merge", merge: mergeDraft, box: resultBox, previousSelection, previousEditRegion, previousGeometrySelection, previousBoxes, previousMerges, priorSourceEdits});
  state.batchReviewBox = resultBox;
  state.editRegion = null;
  state.selection = {...bounds};
  reprojectSelection();
  state.classifierProposal = null;
  restoreBatchLabelDraft(resultBox);
  state.geometryMergeRegions = [];
  state.selectionMergeRegions = [];
  state.mergeRegionIds.clear();
  updateAnnotationSaveAvailability();
  renderBatchBoxes();
  renderRegions();
  updateSelectionMergeControls();
  drawOverlays();
  triggerButton.disabled = false;
  el("merge-status").textContent = `${resultBox.sourceCount} boxes staged as one signal. Label it, then save annotations to apply the merge.`;
  el("geometry-status").textContent = "Merge staged in this 10-second window. Save annotations to apply it.";
}

el("merge-regions").addEventListener("click", async () => {
  const sources = state.regions.filter((region) => state.mergeRegionIds.has(region.id) && region.lineage_state !== "superseded");
  await mergeRegionGroup(sources);
});

el("show-superseded-regions").addEventListener("change", (event) => {
  state.showSupersededRegions = event.currentTarget.checked;
  renderRegions();
});

el("start-scene-review").addEventListener("click", startSceneCoverageReview);
el("start-cv-coverage-queue").addEventListener("click", startCvCoverageQueue);
el("cancel-coverage").addEventListener("click", () => {
  const queueWasActive = Boolean(state.cvCoverageQueue);
  state.cvCoverageQueue = null;
  exitSceneCoverageReview();
  if (queueWasActive) {
    setCvCoverageQueueStatus("Window review stopped. The current window was not saved.");
    el("coverage-status").textContent = "Review stopped. The current window was not saved.";
  }
});
el("mark-coverage-uncertain").addEventListener("click", () => beginCoverageMask("uncertain_regions"));
el("mark-coverage-ignore").addEventListener("click", () => beginCoverageMask("ignore_regions"));
el("cancel-coverage-mask").addEventListener("click", cancelCoverageMask);
el("refresh-coverage").addEventListener("click", () => refreshSceneCoverage());
for (const [id, value] of [["mark-coverage-complete", "complete"], ["mark-coverage-partial", "partial"]]) {
  el(id).addEventListener("click", () => {
    if (el(id).disabled || !state.selection || !state.coverageOnly) return;
    el("coverage-state").value = value;
    el("save-coverage").click();
  });
}
el("save-coverage").addEventListener("click", async () => {
  const captureId = state.capture?.id;
  const selection = state.selection;
  if (!captureId || !selection || state.liveView) return;
  const button = el("save-coverage");
  button.disabled = true;
  el("mark-coverage-complete").disabled = true;
  el("mark-coverage-partial").disabled = true;
  el("coverage-status").textContent = "Saving window review…";
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
    await refreshLiveReviewQueue();
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
      ? "This review changed elsewhere. Refresh window reviews before editing it."
      : `Window review failed: ${error.message}`;
  } finally {
    button.disabled = !state.capture || state.liveView || !state.selection;
    el("mark-coverage-complete").disabled = !state.capture || state.liveView || !state.selection || !state.coverageOnly;
    el("mark-coverage-partial").disabled = !state.capture || state.liveView || !state.selection || !state.coverageOnly;
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

async function deleteSavedRegion(region) {
  if (!state.capture || state.liveView || region.lineage_state === "superseded" || pendingWindowMergeSources().regions.has(region.id)) return;
  saveSelectedBatchLabelDraft();
  const label = region.method || region.family || region.protocol || "Unknown signal";
  if (state.windowRegionDeletes.has(region.id)) {
    state.windowRegionDeletes.delete(region.id);
    state.batchBoxUndo = state.batchBoxUndo.filter((change) => change.type !== "delete-region" || change.regionId !== region.id);
    renderRegions();
    updateAnnotationSaveAvailability();
    el("label-state").textContent = "Signal removal cancelled; the saved region is back in this window draft.";
    return;
  }
  if (!window.confirm(`Remove “${label}” from the active annotations in this 10-second window? Saving keeps its split/merge history when it has lineage.`)) return;
  const previousEdit = state.windowRegionEdits.get(region.id) || null;
  state.batchBoxUndo.push({type: "delete-region", regionId: region.id, previousEdit});
  state.windowRegionEdits.delete(region.id);
  state.windowRegionDeletes.set(region.id, region);
  state.mergeRegionIds.delete(region.id);
  if (state.editRegion?.id === region.id) {
    state.selection = null;
    beginNewRegion();
    el("selection-summary").textContent = "No signal selected";
    el("selection-details").textContent = "Draw boxes for this window, then select one to label and save it.";
  }
  renderRegions();
  updateAnnotationSaveAvailability();
  el("label-state").textContent = "Signal removal staged. Save annotations to apply it, or undo to restore the region.";
}

el("delete-selected-region").addEventListener("click", () => {
  const region = state.regions.find((item) => item.id === state.editRegion?.id);
  if (region) deleteSavedRegion(region);
});

function renderRegions() {
  const list = el("region-list");
  const visibleRegions = state.regions.filter((region) => state.showSupersededRegions || region.lineage_state !== "superseded");
  el("region-count").textContent = String(state.regions.filter((region) => region.lineage_state !== "superseded").length);
  list.replaceChildren();
  if (!visibleRegions.length) {
    const empty = document.createElement("p");
    empty.className = "subhead";
    empty.textContent = state.regions.length
      ? "No active signals. Turn on Show replaced signal history to inspect earlier boxes."
      : "No labeled regions in this recording yet.";
    list.append(empty);
  } else {
    for (const region of visibleRegions) {
      const pendingDelete = state.windowRegionDeletes.has(region.id);
      const pendingEdit = state.windowRegionEdits.get(region.id);
      const pendingMerge = pendingWindowMergeSources().regions.has(region.id);
      const row = document.createElement("div");
      row.className = `${region.lineage_state === "superseded" ? "region-row superseded" : "region-row"}${pendingDelete ? " pending-delete" : pendingEdit ? " pending-edit" : pendingMerge ? " pending-merge" : ""}${state.identityLinkSelection.has(region.id) ? " identity-link-selected" : ""}`;
      row.dataset.regionId = region.id;
      const main = document.createElement("div");
      main.className = "region-main";
      const identityCaption = regionIdentityCaption(region, pendingEdit?.labelDraft || region, pendingEdit?.labelDraft);
      if (identityCaption === "Needs label" && region.lineage_state !== "superseded") row.classList.add("needs-label");
      main.textContent = pendingDelete
        ? `${region.signal_id ? `${region.signal_id} · ` : ""}${identityCaption} · will be removed`
        : pendingMerge
          ? `${region.signal_id ? `${region.signal_id} · ` : ""}${identityCaption} · will merge`
          : `${region.signal_id ? `${region.signal_id} · ` : ""}${identityCaption}`;
      const detail = document.createElement("div");
      detail.className = "region-detail";
      const classifierSummary = classifierEvidenceSummary(region.evidence?.classifier_proposal);
      const detectorEvidence = region.evidence?.detector_proposal;
      const detectorSummary = detectorEvidence
        ? `CV ${Math.round(100 * detectorEvidence.confidence)}% · model ${detectorEvidence.model_sha256.slice(0, 8)}${detectorEvidence.supporting_windows?.length > 1 ? ` · seen in ${detectorEvidence.supporting_windows.length} windows` : ""}`
        : null;
      const displayGeometry = pendingEdit?.geometry || region;
      detail.textContent = `${(displayGeometry.sample_start / state.capture.sample_rate).toFixed(2)} s · ${(displayGeometry.sample_count / state.capture.sample_rate).toFixed(2)} s duration${pendingDelete ? " · removal staged; not yet saved" : pendingMerge ? " · merge staged; save annotations to apply" : pendingEdit ? " · unsaved window edit" : ""}${region.lineage_state === "superseded" ? " · superseded by split/merge" : ""}${detectorSummary ? ` · ${detectorSummary}` : ""}${classifierSummary ? ` · ${classifierSummary}` : ""}`;
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
      const review = document.createElement("button");
      review.type = "button";
      review.textContent = pendingDelete ? "Removal staged" : region.lineage_state === "superseded" ? "History" : "Review";
      review.disabled = pendingDelete || pendingMerge;
      review.addEventListener("click", (event) => {
        if (event.shiftKey && region.lineage_state !== "superseded") {
          if (state.identityLinkSelection.has(region.id)) state.identityLinkSelection.delete(region.id);
          else state.identityLinkSelection.add(region.id);
          refreshIdentityLinkControls();
          renderRegions();
          return;
        }
        selectSavedRegion(region);
      });
      row.append(review);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "danger-button";
      remove.textContent = pendingDelete ? "Undo delete" : "Delete";
      remove.disabled = state.liveView || region.lineage_state === "superseded" || pendingMerge;
      if (region.lineage_state === "superseded") remove.title = "This signal was replaced; remove its active replacement to retain the lineage.";
      else if (region.lineage_related) remove.title = "Remove this active result while retaining its split/merge lineage and label history.";
      remove.setAttribute("aria-label", `Delete ${region.method || region.family || "unknown signal"} region`);
      remove.addEventListener("click", () => deleteSavedRegion(region));
      row.append(remove);
      list.append(row);
    }
  }
  renderRegionShortcuts();
  updateMergeControls();
  refreshIdentityLinkControls();
  drawOverlays();
}

function updateMergeControls() {
  const pendingSources = pendingWindowMergeSources();
  const selected = [...state.mergeRegionIds].filter((regionId) =>
    state.regions.some((region) => region.id === regionId && region.lineage_state !== "superseded")
      && !state.windowRegionDeletes.has(regionId) && !pendingSources.regions.has(regionId),
  );
  state.mergeRegionIds = new Set(selected);
  const button = el("merge-regions");
  el("merge-panel").querySelector("summary").textContent = selected.length
    ? `Select pieces manually · ${selected.length} selected`
    : "Select pieces manually (optional)";
  const maximumReached = selected.length >= 32;
  for (const checkbox of document.querySelectorAll("[data-region-merge-id]")) {
    const region = state.regions.find((item) => item.id === checkbox.dataset.regionMergeId);
    const checked = state.mergeRegionIds.has(checkbox.dataset.regionMergeId);
    checkbox.checked = checked;
    checkbox.disabled = state.liveView || !region || region.lineage_state === "superseded" || state.windowRegionDeletes.has(region?.id) || pendingSources.regions.has(region?.id) || (maximumReached && !checked);
  }
  button.disabled = selected.length < 2 || selected.length > 32 || state.liveView;
  button.textContent = selected.length ? `Merge ${selected.length} selected piece${selected.length === 1 ? "" : "s"}` : "Merge selected pieces";
  el("merge-selection-status").textContent = selected.length === 0
    ? "Select 2–32 saved signal boxes below, regardless of detection source."
    : selected.length === 1
      ? "Select at least one more saved box. CV, classifier, and manual boxes can be combined."
      : `${selected.length} boxes selected · the new signal spans their combined bounds.`;
}

function regionsCrossedByRedraw(bounds, primaryRegionId) {
  const pendingSources = pendingWindowMergeSources();
  const timeEnd = bounds.sample_start + bounds.sample_count;
  return state.regions.filter((region) => {
    if (region.lineage_state === "superseded") return false;
    if (state.windowRegionDeletes.has(region.id)) return false;
    if (pendingSources.regions.has(region.id)) return false;
    if (region.id === primaryRegionId) return true;
    const geometry = state.windowRegionEdits.get(region.id)?.geometry || region;
    const regionEnd = geometry.sample_start + geometry.sample_count;
    const overlapsTime = geometry.sample_start <= timeEnd && regionEnd >= bounds.sample_start;
    if (!overlapsTime) return false;
    // A source with no frequency bounds may still be a duplicate; time overlap
    // is the only reliable intersection available for that source.
    if (!Number.isFinite(geometry.frequency_low_hz) || !Number.isFinite(geometry.frequency_high_hz)) return true;
    return geometry.frequency_low_hz <= bounds.frequency_high_hz
      && geometry.frequency_high_hz >= bounds.frequency_low_hz;
  }).sort((a, b) => a.sample_start - b.sample_start || a.frequency_low_hz - b.frequency_low_hz);
}

function boxesCrossedBySelection(bounds, primaryRegionId = null) {
  const saved = regionsCrossedByRedraw(bounds, primaryRegionId).map((region) => ({ kind: "region", region }));
  const pendingSources = pendingWindowMergeSources();
  const timeEnd = bounds.sample_start + bounds.sample_count;
  const proposals = visibleCvDetections().filter((detection) => {
    const proposalId = detection.cv_proposal_id || detection.proposal_id;
    if (state.selectedTrack?.cv_proposal_id && proposalId === state.selectedTrack.cv_proposal_id) return false;
    if (pendingSources.proposals.has(detection.cv_proposal_id || detection.proposal_id)) return false;
    const source = detection._originalProposalBounds || detection;
    const sourceEnd = source.sample_start + source.sample_count;
    if (source.sample_start > timeEnd || sourceEnd < bounds.sample_start) return false;
    if (!Number.isFinite(source.frequency_low_hz) || !Number.isFinite(source.frequency_high_hz)) return true;
    return source.frequency_low_hz <= bounds.frequency_high_hz
      && source.frequency_high_hz >= bounds.frequency_low_hz;
  }).map((detection) => ({ kind: "cv", detection }));
  const tracks = visibleTracks().filter((track) => {
    if (track === state.selectedTrack) return false;
    if (pendingSources.tracks.has(heuristicCandidateId(track))) return false;
    const trackEnd = track.sample_start + track.sample_count;
    if (track.sample_start > timeEnd || trackEnd < bounds.sample_start) return false;
    if (!Number.isFinite(track.frequency_low_hz) || !Number.isFinite(track.frequency_high_hz)) return true;
    return track.frequency_low_hz <= bounds.frequency_high_hz
      && track.frequency_high_hz >= bounds.frequency_low_hz;
  }).map((track) => ({kind: "track", track}));
  const drafts = state.batchBoxes.filter((box) => {
    const boxEnd = box.sample_start + box.sample_count;
    return box.sample_start <= timeEnd && boxEnd >= bounds.sample_start
      && box.frequency_low_hz <= bounds.frequency_high_hz
      && box.frequency_high_hz >= bounds.frequency_low_hz;
  }).map((box) => ({kind: "draft", box}));
  return [...saved, ...proposals, ...tracks, ...drafts];
}

function heuristicCandidateId(track) {
  return String(track.id || `${track.sample_start}-${track.sample_count}-${track.frequency_low_hz}-${track.frequency_high_hz}`);
}

function mergeCandidateKey(candidate) {
  if (candidate.kind === "cv") return `cv:${candidate.detection.cv_proposal_id || candidate.detection.proposal_id}`;
  if (candidate.kind === "draft") return `draft:${candidate.box.draftMergeId || candidate.box.client_id}`;
  if (candidate.kind === "track") return `track:${heuristicCandidateId(candidate.track)}`;
  return `region:${candidate.region.id}`;
}

function humanMergeIdentity(candidate, labelOverride = null) {
  const item = candidate.kind === "draft" ? candidate.box : candidate.kind === "region" ? candidate.region : null;
  const label = labelOverride || (candidate.kind === "draft"
    ? item?.labelDraft
    : item ? state.windowRegionEdits.get(item.id)?.labelDraft || item : null);
  if (!label) return null;
  const isHuman = candidate.kind === "draft"
    || Boolean(item && (label.source === "human" || state.windowRegionEdits.has(item.id)));
  if (!isHuman) return null;
  const fields = ["mode_id", "family", "method", "protocol", "variant"]
    .map((field) => String(label[field] || "").trim().toLocaleLowerCase())
    .filter(Boolean);
  if (!fields.length) return null;
  const caption = [label.mode_name, label.mode_id, label.variant, label.protocol, label.method, label.family]
    .map((value) => String(value || "").trim())
    .find(Boolean);
  return {key: fields.join("|").replace(/\s+/g, " "), caption};
}

function updateSelectionMergeControls() {
  const actions = el("selection-merge-actions");
  const button = el("merge-selection-boxes");
  const status = el("selection-merge-status");
  const candidateList = el("selection-merge-candidates");
  if (!actions || !button || !status || !candidateList) return;
  const pending = pendingWindowMergeSources();
  const mergeTarget = state.editRegion
    ? state.regions.find((region) => region.id === state.editRegion.id && region.lineage_state === "active")
    : null;
  const current = state.selectionMergeRegions.filter((candidate) => candidate.kind === "cv"
    ? state.cvDetections.includes(candidate.detection) && !pending.proposals.has(candidate.detection.cv_proposal_id || candidate.detection.proposal_id)
    : candidate.kind === "track"
      ? state.tracks.includes(candidate.track) && !pending.tracks.has(heuristicCandidateId(candidate.track))
    : candidate.kind === "draft"
      ? state.batchBoxes.includes(candidate.box) && candidate.box !== state.batchReviewBox
      : state.regions.some((region) => region.id === candidate.region.id
      && region.geometry_revision === candidate.region.geometry_revision
      && region.lineage_state !== "superseded") && !state.windowRegionDeletes.has(candidate.region.id)
      && !pending.regions.has(candidate.region.id) && candidate.region.id !== mergeTarget?.id);
  state.selectionMergeRegions = current;
  actions.hidden = !state.selection || state.liveView || (state.editRegion && !mergeTarget);
  candidateList.replaceChildren();
  for (const candidate of current) {
    const key = mergeCandidateKey(candidate);
    const item = candidate.kind === "cv" ? candidate.detection
      : candidate.kind === "track" ? candidate.track
        : candidate.kind === "draft" ? candidate.box : candidate.region;
    const rate = state.capture?.sample_rate || 1;
    const timeRange = `${(item.sample_start / rate).toFixed(2)}–${((item.sample_start + item.sample_count) / rate).toFixed(2)} s`;
    const title = candidate.kind === "cv"
      ? `CV · ${Math.round((item.confidence || 0) * 100)}% · ${timeRange} · ${(item.frequency_low_hz ?? 0).toFixed(0)}–${(item.frequency_high_hz ?? 0).toFixed(0)} Hz`
      : candidate.kind === "track"
        ? `Heuristic · ${item.modulation_hypothesis || "signal candidate"} · ${timeRange} · ${(item.frequency_low_hz ?? 0).toFixed(0)}–${(item.frequency_high_hz ?? 0).toFixed(0)} Hz`
      : candidate.kind === "draft"
        ? `Draft · ${[item.labelDraft?.mode_name, item.labelDraft?.variant, item.labelDraft?.protocol, item.labelDraft?.method, item.labelDraft?.family].filter(Boolean).join(" · ") || "Unlabeled"} · ${timeRange}`
        : `Saved · ${regionIdentityCaption(item, state.windowRegionEdits.get(item.id)?.labelDraft || item, state.windowRegionEdits.get(item.id)?.labelDraft)} · ${timeRange}`;
    const label = document.createElement("label");
    label.className = "merge-candidate-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = !state.excludedMergeCandidates.has(key);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.excludedMergeCandidates.delete(key);
      else state.excludedMergeCandidates.add(key);
      updateSelectionMergeControls();
      drawOverlays();
    });
    const text = document.createElement("span");
    text.textContent = title;
    label.append(checkbox, text);
    candidateList.append(label);
  }
  const included = current.filter((candidate) => !state.excludedMergeCandidates.has(mergeCandidateKey(candidate)));
  const candidateLimit = mergeTarget || state.selectedTrack ? 31 : 32;
  button.disabled = included.length < 1 || included.length > candidateLimit;
  const humanIdentities = [];
  let targetResultLabel;
  let targetUnknownConfirmed = false;
  if (mergeTarget) {
    const target = state.regions.find((region) => region.id === mergeTarget.id);
    targetResultLabel = target && (state.windowRegionEdits.get(target.id)?.labelDraft || target);
    targetUnknownConfirmed = Boolean(targetResultLabel?.identity_unknown_confirmed
      ?? target?.evidence?.identity_unknown_confirmed);
    const identity = target && humanMergeIdentity({kind: "region", region: target}, targetResultLabel);
    if (identity) humanIdentities.push(identity);
  } else {
    targetResultLabel = readBatchLabelDraft();
    targetUnknownConfirmed = Boolean(targetResultLabel.identity_unknown_confirmed);
    const targetDraft = state.batchReviewBox || {labelDraft: targetResultLabel};
    const identity = humanMergeIdentity({kind: "draft", box: targetDraft}, targetResultLabel);
    if (identity) humanIdentities.push(identity);
  }
  for (const candidate of included) {
    const identity = humanMergeIdentity(candidate);
    if (identity) humanIdentities.push(identity);
  }
  const distinctIdentities = [...new Map(humanIdentities.map((identity) => [identity.key, identity])).values()];
  const targetLabel = mergeTarget || state.batchReviewBox || state.selectedTrack
    ? "into this signal"
    : "into this selection";
  const targetLabelFields = ["mode_name", "mode_id", "family", "method", "protocol", "variant"]
    .map((field) => String(targetResultLabel?.[field] || "").trim())
    .filter(Boolean);
  const targetLabelCaption = targetLabelFields[0];
  const mergeResolution = targetLabelCaption
    ? `The selected label ${targetLabelCaption} will stay on the result.`
    : targetUnknownConfirmed
      ? "The result will keep its explicitly confirmed Unknown identity."
      : "No source label will be chosen automatically; label the merged signal before saving.";
  button.textContent = included.length > candidateLimit
    ? `Selection includes ${included.length} boxes`
    : !current.length
      ? "Merge intersecting boxes (0)"
      : `Merge ${included.length} intersecting ${included.length === 1 ? "box" : "boxes"} ${targetLabel}`;
  status.textContent = distinctIdentities.length > 1
    ? `Conflicting human labels: ${distinctIdentities.map((identity) => identity.caption).join(" vs ")}. ${mergeResolution} Exclude boxes that belong to another signal.`
    : included.length > candidateLimit
    ? "Merge up to 32 boxes at a time. Exclude some boxes to continue."
    : !current.length
      ? "No other boxes intersect this selection. Adjust its bounds or select another signal."
      : `${included.length} included · ${current.length - included.length} excluded. Review which boxes belong to this signal.`;
}

el("merge-selection-boxes").addEventListener("click", async () => {
  const pending = pendingWindowMergeSources();
  const crossed = state.selectionMergeRegions.filter((candidate) => candidate.kind === "cv"
    ? state.cvDetections.includes(candidate.detection)
      && !pending.proposals.has(candidate.detection.cv_proposal_id || candidate.detection.proposal_id)
      && !state.excludedMergeCandidates.has(mergeCandidateKey(candidate))
    : candidate.kind === "track"
      ? state.tracks.includes(candidate.track)
        && !pending.tracks.has(heuristicCandidateId(candidate.track))
        && !state.excludedMergeCandidates.has(mergeCandidateKey(candidate))
    : candidate.kind === "draft"
      ? state.batchBoxes.includes(candidate.box)
        && !state.excludedMergeCandidates.has(mergeCandidateKey(candidate))
      : state.regions.some((region) => region.id === candidate.region.id
      && region.geometry_revision === candidate.region.geometry_revision
      && region.lineage_state !== "superseded")
      && !pending.regions.has(candidate.region.id)
      && !state.excludedMergeCandidates.has(mergeCandidateKey(candidate)));
  const selected = state.regions.find((region) => region.id === state.editRegion?.id && region.lineage_state === "active");
  const selectedCvId = state.selectedTrack?.cv_proposal_id || state.selectedTrack?.proposal_id;
  const selectedCvDetection = selectedCvId && state.cvDetections.find((detection) =>
    (detection.cv_proposal_id || detection.proposal_id) === selectedCvId);
  const selectedTrackSource = selectedCvDetection
    ? {kind: "cv", detection: selectedCvDetection}
    : state.selectedTrack && state.tracks.includes(state.selectedTrack)
      ? {kind: "track", track: state.selectedTrack}
      : null;
  const selectedSources = selected
    ? [{kind: "region", region: selected}]
    : selectedTrackSource ? [selectedTrackSource] : [];
  const sources = [...selectedSources, ...crossed];
  const selection = state.selection;
  if (sources.length < (selectedSources.length ? 2 : 1) || sources.length > 32 || !selection) return;
  await mergeRegionGroup(sources, {
    sample_start: selection.sample_start,
    sample_count: selection.sample_count,
    frequency_low_hz: selection.frequency_low_hz,
    frequency_high_hz: selection.frequency_high_hz,
  }, el("merge-selection-boxes"));
});

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
  if (tab !== "signals" && state.coverageOnly) exitSceneCoverageReview();
  if (tab === "decode" || tab === "raw") setInspectorMode("classify");
  if (state.inspectorMode === "classify") {
    state.inspectorTab = tab;
    try { localStorage.setItem("sig-id-inspector-tab", tab); } catch {}
  }
  el("inspector").classList.remove("hidden");
  syncInspectorModeUI();
  updateDecoderSelectionControls();
}
function syncInspectorModeUI() {
  const annotate = state.inspectorMode === "annotate";
  el("inspector").dataset.workflowMode = state.inspectorMode;
  el("output-dock").dataset.workflowMode = state.inspectorMode;
  el("inspector-tabs").hidden = annotate;
  for (const node of document.querySelectorAll("[data-inspector-view]")) {
    node.hidden = annotate
      ? !["label", "signals"].includes(node.dataset.inspectorView)
      : node.dataset.inspectorView !== state.inspectorTab;
  }
  for (const button of document.querySelectorAll("[data-inspector-tab]")) button.setAttribute("aria-pressed", String(!annotate && button.dataset.inspectorTab === state.inspectorTab));
  for (const button of document.querySelectorAll("[data-inspector-mode]")) button.setAttribute("aria-pressed", String(button.dataset.inspectorMode === state.inspectorMode));
  el("save-annotations").hidden = annotate;
  el("save-annotations-close").hidden = !annotate;
  updateAnnotationSaveAvailability();
}
function setInspectorMode(mode) {
  if (mode !== "classify" && mode !== "annotate") return;
  saveSelectedBatchLabelDraft();
  state.inspectorMode = mode;
  el("inspector").dataset.workflowMode = mode;
  try { localStorage.setItem("sig-id-inspector-mode", mode); } catch {}
  if (mode === "annotate") {
    state.inspectorTab = "label";
    state.batchBoxMode = true;
    beginNewRegion();
  } else state.batchBoxMode = false;
  syncInspectorModeUI();
  openInspector(mode === "annotate" ? "label" : state.inspectorTab);
}
function setInspectorDetailLevel(level) {
  if (level !== "simple" && level !== "advanced") return;
  // Keep an in-progress coverage form and its exit controls reachable.
  if (level === "simple" && state.coverageOnly) level = "advanced";
  state.inspectorDetailLevel = level;
  el("inspector").dataset.detailLevel = level;
  try { localStorage.setItem("sig-id-inspector-detail", level); } catch {}
  for (const button of document.querySelectorAll("[data-detail-level]")) {
    button.setAttribute("aria-pressed", String(button.dataset.detailLevel === level));
  }
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
  if (el("auto-identify-selection")) {
    el("auto-identify-selection").disabled = !(validSavedSelection || validLiveSelection)
      || !(state.fldigiModes?.length);
  }
  if (el("play-selection")) {
    el("play-selection").disabled = !selectedAudioPlayback && !validSavedSelection;
    if (!selectedAudioPlayback) el("play-selection").textContent = "Replay selected region";
  }
  if (el("tag-as-mode")) el("tag-as-mode").disabled = !validSavedSelection || !validMode;
  if (el("native-decode-selection")) {
    const selectedNativeMode = nativeDecoderBindings.some((mode) => `${mode.backend_id}|${mode.mode_id}` === el("native-decoder-mode")?.value);
    el("native-decode-selection").disabled = !validSavedSelection || !selectedNativeMode;
  }
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
  for (const [index, track] of visibleTracks().entries()) {
    const start = Math.min(sampleToRow(track.sample_start), sampleToRow(track.sample_start + track.sample_count));
    const end = Math.max(sampleToRow(track.sample_start), sampleToRow(track.sample_start + track.sample_count));
    if (end < 0 || start >= overlayCanvas.height || track.frequency_high_hz < low || track.frequency_low_hz > high) continue;
    const trial = state.decoderTrial?.track_id === track.track_id ? state.decoderTrial : null;
    const activity = trial?.status === "running" ? ` · trying ${trial.mode}` : trial?.status === "tentative_output" ? ` · tentative ${trial.mode} output` : "";
    const trackDetail = [
      Number.isFinite(track.peak_snr_db) ? `${track.peak_snr_db.toFixed(1)} dB SNR` : null,
      track.tone_count_candidate ? `${track.tone_count_candidate} tones` : null,
      Number.isFinite(track.tone_spacing_hz_candidate) ? `${track.tone_spacing_hz_candidate.toFixed(1)} Hz spacing` : null,
    ].filter(Boolean).join(" · ");
    items.push({
      track, text: `${signalCaption(track, index)}${activity}${trackDetail ? `\n${trackDetail}` : ""}`,
      selected: state.selectedTrack === track || Boolean(track.track_id && state.selectedTrack?.track_id === track.track_id),
      x: (track.frequency_low_hz - low) / Math.max(1, high - low) * width,
      boxRight: (track.frequency_high_hz - low) / Math.max(1, high - low) * width,
      boxCenterY: (start + end) / 2 / overlayCanvas.height * height,
      y: Math.max(0, start / overlayCanvas.height * height),
    });
  }
  for (const [index, detection] of visibleCvDetections().entries()) {
    const start = Math.min(sampleToRow(detection.sample_start), sampleToRow(detection.sample_start + detection.sample_count));
    const end = Math.max(sampleToRow(detection.sample_start), sampleToRow(detection.sample_start + detection.sample_count));
    if (end < 0 || start >= overlayCanvas.height || detection.frequency_high_hz < low || detection.frequency_low_hz > high) continue;
    items.push({
      cvDetection: detection,
      text: `CV${index + 1} · ${(100 * (detection.confidence || 0)).toFixed(0)}% detector\n${state.liveView ? cvClassifierCaption(detection) : `Needs label${detection.classifier_proposal ? ` · ${cvClassifierCaption(detection)}` : ""}`}`,
      needsLabel: !state.liveView,
      selected: Boolean(detection.cv_proposal_id && state.selectedTrack?.cv_proposal_id === detection.cv_proposal_id),
      x: (detection.frequency_low_hz - low) / Math.max(1, high - low) * width,
      boxRight: (detection.frequency_high_hz - low) / Math.max(1, high - low) * width,
      boxCenterY: (start + end) / 2 / overlayCanvas.height * height,
      y: Math.max(0, start / overlayCanvas.height * height),
    });
  }
  for (const region of state.regions) {
    // Superseded pieces remain in lineage history but must not leave duplicate labels on the waterfall.
    if (region.lineage_state === "superseded") continue;
    if (state.windowRegionDeletes.has(region.id)) continue;
    const draft = state.windowRegionEdits.get(region.id);
    const geometry = draft?.geometry || region;
    const label = draft?.labelDraft || region;
    const start = Math.min(sampleToRow(geometry.sample_start), sampleToRow(geometry.sample_start + geometry.sample_count));
    const end = Math.max(sampleToRow(geometry.sample_start), sampleToRow(geometry.sample_start + geometry.sample_count));
    if (end < 0 || start >= overlayCanvas.height || (geometry.frequency_high_hz ?? high) < low || (geometry.frequency_low_hz ?? low) > high) continue;
    const identityCaption = regionIdentityCaption(region, label, draft?.labelDraft);
    items.push({region, text: `${region.signal_id ? `${region.signal_id} · ` : ""}${draft ? "Draft · " : region.lineage_state === "superseded" ? "Superseded · " : "Saved · "}${identityCaption}`,
      needsLabel: identityCaption === "Needs label",
      superseded: region.lineage_state === "superseded",
      selected: state.identityLinkSelection.has(region.id),
      x: ((geometry.frequency_low_hz ?? low) - low) / Math.max(1, high - low) * width,
      boxRight: ((geometry.frequency_high_hz ?? high) - low) / Math.max(1, high - low) * width,
      boxCenterY: (start + end) / 2 / overlayCanvas.height * height,
      y: Math.max(0, start / overlayCanvas.height * height)});
  }
  for (const [index, box] of state.batchBoxes.entries()) {
    const start = Math.min(sampleToRow(box.sample_start), sampleToRow(box.sample_start + box.sample_count));
    const end = Math.max(sampleToRow(box.sample_start), sampleToRow(box.sample_start + box.sample_count));
    if (end < 0 || start >= overlayCanvas.height || box.frequency_high_hz < low || box.frequency_low_hz > high) continue;
    const draftLabel = box.labelDraft || {};
    const needsLabel = !draftLabel.identity_unknown_confirmed
      && ![draftLabel.mode_id, draftLabel.family, draftLabel.method, draftLabel.protocol, draftLabel.variant].some(Boolean);
    items.push({draftBox: box, draftIndex: index, text: batchBoxLabelText(box, index), needsLabel,
      selected: state.batchReviewBox === box,
      x: (box.frequency_low_hz - low) / Math.max(1, high - low) * width,
      boxRight: (box.frequency_high_hz - low) / Math.max(1, high - low) * width,
      boxCenterY: (start + end) / 2 / overlayCanvas.height * height,
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
      line.setAttribute("x2", String(item.leaderToX));
      line.setAttribute("y2", String(item.leaderToY));
      leaders.append(line);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `signal-tag ${item.region ? "saved" : item.cvDetection ? "cv" : "candidate"}${item.needsLabel ? " needs-label" : ""}${item.superseded ? " superseded" : ""}${item.selected ? " selected" : ""}`;
    button.textContent = item.text;
    button.style.left = `${item.x}px`;
    button.style.top = `${item.y}px`;
    button.style.width = `${item.width}px`;
    button.style.height = `${item.height}px`;
    button.dataset.signalKey = item.track?.track_id || item.region?.id || item.cvDetection?.cv_proposal_id || item.draftBox?.client_id || item.text;
    button.title = item.cvDetection
      ? `${item.text} · ${(item.cvDetection.confidence * 100).toFixed(1)}% detector score\n${item.cvDetection.frequency_low_hz.toFixed(0)}–${item.cvDetection.frequency_high_hz.toFixed(0)} Hz · ${(item.cvDetection.sample_start / data.sample_rate).toFixed(2)}–${((item.cvDetection.sample_start + item.cvDetection.sample_count) / data.sample_rate).toFixed(2)} s\nClick to select this provisional CV box and review its identity.`
      : item.draftBox
      ? `Draft signal · ${item.text}\n${item.draftBox.frequency_low_hz.toFixed(0)}–${item.draftBox.frequency_high_hz.toFixed(0)} Hz · ${(item.draftBox.sample_start / data.sample_rate).toFixed(2)}–${((item.draftBox.sample_start + item.draftBox.sample_count) / data.sample_rate).toFixed(2)} s\nClick to select this annotation draft.`
      : item.track
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
    button.addEventListener("click", (event) => {
      if (item.region && event.shiftKey) {
        if (state.identityLinkSelection.has(item.region.id)) state.identityLinkSelection.delete(item.region.id);
        else state.identityLinkSelection.add(item.region.id);
        refreshIdentityLinkControls();
        renderRegions();
        return;
      }
      item.draftBox
      ? reviewBatchBox(item.draftIndex)
      : item.cvDetection
      ? selectCvDetection(item.cvDetection)
      : item.track ? selectTrack(item.track) : selectSavedRegion(item.region);
    });
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
function selectSavedRegion(region, inspectorView = "label") {
  if (state.windowRegionDeletes.has(region.id)) return;
  stopSelectedAudioPlayback(null);
  saveSelectedBatchLabelDraft();
  if (state.editRegion?.id !== region.id) saveSelectedExistingRegionDraft();
  const pendingDraft = state.windowRegionEdits.get(region.id) || null;
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
  const visibleGeometry = pendingDraft?.geometry || region;
  state.selection = {sample_start:visibleGeometry.sample_start, sample_count:visibleGeometry.sample_count,
    frequency_low_hz:visibleGeometry.frequency_low_hz ?? 0, frequency_high_hz:visibleGeometry.frequency_high_hz ?? Math.min(3000,data.sample_rate/2)};
  state.coverageEdit = null;
  state.coverageOnly = false;
  el("coverage-panel").hidden = true;
  el("save-coverage").textContent = "Record reviewed window";
  updateSceneReviewAvailability();
  updateCarrierFromSelection(state.selection);
  state.selectedTrack = null;
  state.batchReviewBox = null;
  el("batch-unknown-row").hidden = true;
  el("batch-unknown-confirmed").checked = false;
  state.editRegion = {id:region.id, revision:region.revision, geometry_revision:region.geometry_revision};
  state.selectionMergeRegions = boxesCrossedBySelection(state.selection, region.id);
  updateSelectionMergeControls();
  state.geometryMergeRegions = [];
  el("merge-redrawn-regions").hidden = true;
  el("geometry-merge-status").hidden = true;
  el("geometry-merge-status").textContent = "";
  renderRegionShortcuts();
  setModeProposalSource(region);
  state.labelModeId = pendingDraft?.labelDraft.mode_id ?? region.mode_id ?? null;
  el("mode-search").value = pendingDraft?.labelDraft.mode_name ?? region.mode_id ?? "";
  el("batch-unknown-row").hidden = false;
  el("batch-unknown-confirmed").checked = pendingDraft?.labelDraft.identity_unknown_confirmed ?? Boolean(region.evidence?.identity_unknown_confirmed);
  state.modeSearchRequest += 1;
  el("mode-results").replaceChildren();
  el("mode-resolution").textContent = region.mode_id ? `Canonical mode ${region.mode_id} · taxonomy retained with registry version in label evidence` : "Partial labels are welcome; unknown attributes stay blank.";
  el("missed-signal").checked = region.origin === "user_added_missed_detection";
  el("feedback-panel").hidden = region.origin !== "detector_proposal" || region.lineage_state === "superseded";
  el("feedback-status").textContent = "";
  el("bounds-panel").hidden = false;
  const signalName = [region.family, region.method, region.protocol, region.variant].filter(Boolean).join(" · ") || "Unknown signal";
  const startSeconds = region.sample_start / data.sample_rate;
  const endSeconds = (region.sample_start + region.sample_count) / data.sample_rate;
  el("signal-extent-title").textContent = `${signalName} · ${startSeconds.toFixed(2)}–${endSeconds.toFixed(2)} s`;
  el("edit-region-bounds").disabled = region.lineage_state === "superseded";
  el("delete-selected-region").disabled = state.liveView || region.lineage_state === "superseded";
  el("delete-selected-region").title = region.lineage_state === "superseded"
    ? "This signal was replaced; remove its active replacement to retain the lineage."
    : region.lineage_related
      ? "Remove this active result while retaining its split/merge lineage and label history."
    : "Stage removal in this 10-second window; Save annotations to apply it or Undo delete to restore it.";
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
  for (const field of ["family","method","protocol","variant","notes"]) el(field).value = pendingDraft?.labelDraft[field] ?? region[field] ?? "";
  el("selection-summary").textContent = region.variant || region.protocol || region.method || region.family || "Saved unknown region";
  el("selection-details").textContent = `${region.source} · ${region.review_state} · ${(region.sample_start/data.sample_rate).toFixed(2)}–${((region.sample_start+region.sample_count)/data.sample_rate).toFixed(2)} s · drag inside to move or a corner to resize`;
  el("label-state").textContent = `Editing label revision ${region.revision} · original history retained`;
  el("save-region").textContent = "Save revision";
  el("inspector-context").textContent = `Saved annotation · ${state.selection.frequency_low_hz.toFixed(0)}–${state.selection.frequency_high_hz.toFixed(0)} Hz · revision ${region.revision}`;
  loadLabelHistory(region, state.capture.id);
  loadGeometryHistory(region, state.capture.id);
  loadLineageHistory(region, state.capture.id);
  setRegionSaveDisabled(state.liveView || region.lineage_state === "superseded");
  updateModelButton();
  for (const dialog of document.querySelectorAll("dialog[open]")) dialog.close();
  openInspector(inspectorView);
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
async function reviewQueuedCapture(capture) {
  el("review-queue-dialog").close();
  state.capture = capture;
  try {
    const loaded = await loadCapture();
    if (!loaded) {
      el("review-queue-dialog").showModal();
      return;
    }
    openInspector("signals");
    if (state.cvAvailable && !restoreCachedCvDetection(capture)) {
      try {
        const result = await request(`/api/captures/${encodeURIComponent(capture.id)}/cv-detections`, { method: "POST" });
        if (state.capture !== capture) return;
        applyCvDetectionResult(capture, result);
      } catch (error) {
        el("cv-status").textContent = `Could not scan queued sample · ${error.message}`;
      }
    }
    el("box-draw-status").textContent = "Sample ready · draw a box around each signal, or use multiple selections. Use Advanced → Window coverage when you need to review a specific time/frequency window.";
  } catch (error) {
    showError(`Could not open queued sample · ${error.message}`);
  }
}

async function reviewRandomDatasetWindow() {
  if (state.liveView) {
    showError("Stop the live capture before opening a random saved window.");
    return;
  }
  const button = el("review-random-dataset");
  button.disabled = true;
  const label = button.textContent;
  button.textContent = "Choosing a window…";
  try {
    const captures = await request("/api/captures");
    const eligible = captures.filter((capture) => capture.source_kind !== "synthetic"
      && capture.source_metadata?.auto_review_queue === true
      && Number(capture.source_metadata?.window_seconds) === ANNOTATION_WINDOW_SECONDS
      && capture.sample_rate > 0
      && capture.sample_count === Math.round(capture.sample_rate * ANNOTATION_WINDOW_SECONDS)
      && capture.source_metadata?.exclude_training !== true);
    const windows = eligible.map((capture) => ({capture, index: 0}));
    if (!windows.length) {
      showError("There are no saved 10-second review samples available yet.");
      return;
    }
    const currentCapture = state.loadedCapture;
    let choices = windows;
    if (windows.length > 1 && currentCapture) {
      const alternatives = windows.filter((item) => item.capture.id !== currentCapture.id
        || item.index !== state.annotationWindowIndex);
      if (alternatives.length) choices = alternatives;
    }
    const picked = choices[Math.floor(Math.random() * choices.length)];
    if (picked.capture.id === currentCapture?.id
        && !await resolveWindowDraftsBeforeSwitch("opening a random dataset window")) return;
    state.capture = picked.capture;
    const loaded = await loadCapture({windowIndex: picked.index, focusedWindow: true});
    if (!loaded) return;
    updateAnnotationWindowView();
    setInspectorMode("annotate");
    openInspector("label");
    el("inspector-context").textContent = `Random dataset review · window ${picked.index + 1} of ${Math.ceil(picked.capture.sample_count / (picked.capture.sample_rate * ANNOTATION_WINDOW_SECONDS))}`;
  } catch (error) {
    showError(`Could not open a random dataset window · ${error.message}`);
  } finally {
    button.textContent = label;
    button.disabled = false;
  }
}

async function refreshLiveReviewQueue() {
  try {
    const result = await request("/api/live/review-queue");
    const badge = el("review-queue-badge");
    badge.textContent = String(result.pending_count);
    badge.hidden = result.pending_count === 0;
    el("review-queue-title-count").textContent = `${result.pending_count} waiting · ${result.reviewed_count} reviewed · ${result.re_review_due_count || 0} due again`;
    const list = el("review-queue-list");
    list.replaceChildren();
    for (const capture of result.pending) {
      const row = document.createElement("div");
      row.className = "review-queue-row";
      const info = document.createElement("div");
      const title = document.createElement("strong");
      const trigger = capture.source_metadata.review_trigger === "cv_signal"
        ? "CV signal candidate"
        : capture.source_metadata.review_trigger === "periodic_noise"
          ? "Periodic noise candidate"
          : capture.source_metadata.created_from_live_annotation ? "Classify sample" : "Live sample";
      title.textContent = `${capture.source_metadata.device_name || "Live input"} · ${trigger} · ${(capture.source_metadata.window_start_sample / capture.sample_rate).toFixed(1)} s${capture.source_metadata.re_review_due ? " · re-review due" : ""}`;
      const detail = document.createElement("div");
      detail.className = "muted";
      const detection = capture.source_metadata.detection_summary || {};
      const detectorDetail = capture.source_metadata.unattended_capture
        ? ` · ${detection.cv_detection_count || 0} CV boxes${detection.classifier_methods?.length ? ` · ${[...new Set(detection.classifier_methods)].join(", ")}` : ""}`
        : "";
      detail.textContent = `${capture.sample_rate.toLocaleString()} samples/s · ${capture.sample_count.toLocaleString()} samples${detectorDetail} · ${capture.id.slice(0, 8)}${capture.source_metadata.re_review_due ? ` · last reviewed ${new Date(capture.source_metadata.last_reviewed_utc).toLocaleDateString()}` : ""}`;
      info.append(title, detail);
      const open = document.createElement("button");
      open.type = "button";
      open.textContent = "Review";
      open.disabled = state.liveView;
      open.title = state.liveView ? "Stop live capture before opening a queued sample" : "Open sample to annotate signals";
      open.addEventListener("click", () => reviewQueuedCapture(capture));
      const markReviewed = document.createElement("button");
      markReviewed.type = "button";
      markReviewed.textContent = "Mark reviewed";
      markReviewed.title = "Remove this 10-second window from the live review queue without recording coverage";
      markReviewed.addEventListener("click", async () => {
        markReviewed.disabled = true;
        try {
          await request(`/api/live/review-queue/${encodeURIComponent(capture.id)}/mark-reviewed`, { method: "POST" });
          await refreshLiveReviewQueue();
        } catch (error) {
          showError(`Could not mark sample reviewed · ${error.message}`);
          markReviewed.disabled = false;
        }
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "Delete recording";
      remove.className = "danger-button";
      remove.title = "Permanently remove this queued sample and its audio";
      remove.addEventListener("click", async () => {
        const name = capture.source_metadata.device_name || "Live sample";
        if (!window.confirm(`Permanently delete this ${name} review recording and its audio? This cannot be undone.`)) return;
        remove.disabled = true;
        try {
          await request(`/api/captures/${encodeURIComponent(capture.id)}`, { method: "DELETE" });
          await refreshLiveReviewQueue();
        } catch (error) {
          showError(`Could not delete review recording · ${error.message}`);
          remove.disabled = false;
        }
      });
      const actions = document.createElement("div");
      actions.className = "review-queue-actions";
      actions.append(open, markReviewed, remove);
      row.append(info, actions);
      list.append(row);
    }
    if (!result.pending.length) list.textContent = "No live samples are waiting for review.";
    const reviewedList = el("reviewed-queue-list");
    reviewedList.replaceChildren();
    const manuallyReviewed = result.manually_reviewed || [];
    el("reviewed-queue-details").hidden = !manuallyReviewed.length;
    el("reviewed-queue-summary").textContent = `Manually reviewed · ${manuallyReviewed.length} · return to queue`;
    for (const capture of manuallyReviewed) {
      const row = document.createElement("div");
      row.className = "review-queue-row";
      const info = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${capture.source_metadata.device_name || "Live sample"} · ${(capture.source_metadata.window_start_sample / capture.sample_rate).toFixed(1)} s`;
      const detail = document.createElement("div");
      detail.className = "muted";
      detail.textContent = `Marked reviewed ${new Date(capture.source_metadata.auto_review_queue_reviewed_utc).toLocaleString()} · ${capture.id.slice(0, 8)}`;
      info.append(title, detail);
      const reopen = document.createElement("button");
      reopen.type = "button";
      reopen.textContent = "Return to queue";
      reopen.addEventListener("click", async () => {
        reopen.disabled = true;
        try {
          await request(`/api/live/review-queue/${encodeURIComponent(capture.id)}/reopen`, { method: "POST" });
          await refreshLiveReviewQueue();
        } catch (error) {
          showError(`Could not return sample to review queue · ${error.message}`);
          reopen.disabled = false;
        }
      });
      const actions = document.createElement("div");
      actions.className = "review-queue-actions";
      actions.append(reopen);
      row.append(info, actions);
      reviewedList.append(row);
    }
  } catch (error) {
    el("review-queue-list").textContent = `Could not load review queue · ${error.message}`;
  }
}

async function refreshHubReviewQueue() {
  const list = el("hub-review-queue-list");
  const status = el("hub-review-status");
  list.replaceChildren();
  status.textContent = "Loading Hub review queue…";
  try {
    const items = await request("/api/hub/review-queue");
    status.textContent = `${items.length} Hub annotation${items.length === 1 ? "" : "s"} · Hub reviews assess signal extent; they are not local training approval.`;
    for (const item of items) {
      const row = document.createElement("article");
      row.className = "hub-review-row";
      const title = document.createElement("strong");
      title.textContent = `Hub contribution · ${item.submission_id.slice(0, 8)} · ${item.consensus.replaceAll("_", " ")}`;
      const annotation = item.annotation;
      const bounds = document.createElement("p");
      const sampleRate = item.manifest.sample_rate;
      const seconds = `${(annotation.sample_start / sampleRate).toFixed(2)}–${((annotation.sample_start + annotation.sample_count) / sampleRate).toFixed(2)} s`;
      const label = [annotation.mode_id, annotation.variant, annotation.protocol, annotation.method, annotation.family].find(Boolean) || "Identity unknown";
      bounds.textContent = `${label} · ${seconds} · ${annotation.frequency_low_hz ?? "?"}–${annotation.frequency_high_hz ?? "?"} Hz · ${annotation.completeness.replaceAll("_", " ")}`;
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `/api/hub/submissions/${encodeURIComponent(item.submission_id)}/audio`;
      const controls = document.createElement("div");
      controls.className = "hub-review-controls";
      const decision = document.createElement("select");
      for (const value of ["looks_right", "adjust_bounds", "multiple_signals", "not_signal", "unsure"]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value.replaceAll("_", " ");
        decision.append(option);
      }
      const note = document.createElement("input");
      note.type = "text";
      note.maxLength = 2000;
      note.placeholder = "Optional review note";
      const submit = document.createElement("button");
      submit.type = "button";
      submit.textContent = "Submit Hub review";
      submit.addEventListener("click", async () => {
        submit.disabled = true;
        try {
          const result = await request(`/api/hub/submissions/${encodeURIComponent(item.submission_id)}/annotations/${encodeURIComponent(annotation.annotation_id)}/reviews`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ revision: annotation.revision, decision: decision.value, note: note.value.trim() }),
          });
          status.textContent = `Hub review recorded · ${result.consensus.replaceAll("_", " ")}`;
          await refreshHubReviewQueue();
        } catch (error) {
          showError(`Could not submit Hub review · ${error.message}`);
          submit.disabled = false;
        }
      });
      controls.append(decision, note, submit);
      row.append(title, bounds, audio, controls);
      list.append(row);
    }
    if (!items.length) list.textContent = "The Hub review queue is empty.";
  } catch (error) {
    status.textContent = `Hub queue unavailable · ${error.message}`;
  }
}

for (const button of document.querySelectorAll("[data-review-source]")) {
  button.addEventListener("click", () => {
    const hub = button.dataset.reviewSource === "hub";
    el("local-review-source").hidden = hub;
    el("hub-review-source").hidden = !hub;
    for (const sourceButton of document.querySelectorAll("[data-review-source]")) {
      sourceButton.setAttribute("aria-pressed", String(sourceButton === button));
    }
    if (hub) refreshHubReviewQueue();
  });
}
for (const button of document.querySelectorAll("[data-inspector-mode]")) button.addEventListener("click", () => setInspectorMode(button.dataset.inspectorMode));
for (const button of document.querySelectorAll("[data-detail-level]")) button.addEventListener("click", () => setInspectorDetailLevel(button.dataset.detailLevel));
for (const button of document.querySelectorAll("[data-inspector-tab]")) button.addEventListener("click", () => openInspector(button.dataset.inspectorTab));
for (const button of document.querySelectorAll("[data-output-tab]")) button.addEventListener("click", () => showOutput(button.dataset.outputTab));
for (const button of document.querySelectorAll("[data-close-dialog]")) button.addEventListener("click", () => button.closest("dialog").close());
el("show-signals").addEventListener("click", () => openInspector("signals"));
el("close-inspector").addEventListener("click", () => el("inspector").classList.add("hidden"));
el("show-settings").addEventListener("click", () => el("settings-dialog").showModal());
for (const id of ["show-library", "open-recording"]) el(id).addEventListener("click", () => { refreshLibrary(); el("library-dialog").showModal(); });
el("show-review-queue").addEventListener("click", () => { refreshLiveReviewQueue(); el("review-queue-dialog").showModal(); });
el("review-random-dataset").addEventListener("click", reviewRandomDatasetWindow);
if (state.inspectorMode === "annotate") state.batchBoxMode = true;
syncInspectorModeUI();
setInspectorDetailLevel(state.inspectorDetailLevel);
refreshLiveReviewQueue();
setInterval(refreshLiveReviewQueue, 15000);
refreshNativeDecoderStatus();
setInterval(refreshNativeDecoderStatus, 3000);
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
function clearCurrentSelection(message = "Selection cleared.") {
  saveSelectedBatchLabelDraft();
  state.selection = null;
  state.classifierProposal = null;
  state.selectedTrack = null;
  state.batchReviewBox = null;
  beginNewRegion();
  el("selection-summary").textContent = "No selection";
  el("selection-details").textContent = "Drag over the waterfall to mark a signal.";
  el("inspector-context").textContent = message;
  el("label-state").textContent = message;
  drawOverlays();
  updateModelButton();
}

async function deleteCurrentSelection() {
  if (state.liveView) return;
  saveSelectedBatchLabelDraft();
  if (state.editRegion) {
    const region = state.regions.find((item) => item.id === state.editRegion.id);
    if (region?.lineage_state === "superseded") {
      el("label-state").textContent = "This signal was replaced. Remove its active replacement to keep the lineage history.";
    } else if (region) await deleteSavedRegion(region);
    return;
  }
  if (state.selectedTrack?.cv_proposal_id) {
    if (pendingWindowMergeSources().proposals.has(state.selectedTrack.cv_proposal_id)) {
      el("label-state").textContent = "This proposal is staged in a merge. Undo the merge draft before rejecting the detector box.";
      return;
    }
    const detection = state.cvDetections.find((item) =>
      (item.cv_proposal_id || item.proposal_id) === state.selectedTrack.cv_proposal_id);
    if (detection) await rejectCvDetection(detection);
    return;
  }
  if (state.batchReviewBox) {
    const box = state.batchReviewBox;
    const index = state.batchBoxes.indexOf(box);
    if (box.draftMergeId) {
      const undoIndex = state.batchBoxUndo.findLastIndex((change) => change.type === "undo-merge" && change.box === box);
      const stagedUndo = undoIndex >= 0 ? state.batchBoxUndo.splice(undoIndex, 1)[0] : null;
      state.batchBoxUndo = state.batchBoxUndo.filter((change) => change.box !== box);
      if (stagedUndo) {
        state.batchBoxes = stagedUndo.previousBoxes;
        state.windowMerges = stagedUndo.previousMerges;
        for (const edit of stagedUndo.priorSourceEdits || []) state.windowRegionEdits.set(edit.region_id, edit);
        state.editRegion = stagedUndo.previousEditRegion || null;
        state.selection = stagedUndo.previousGeometrySelection ? {...stagedUndo.previousGeometrySelection} : null;
      } else state.batchBoxes.splice(index, 1);
    } else {
      state.batchBoxUndo.push({type: "remove", box, index});
      state.batchBoxes.splice(index, 1);
    }
    clearCurrentSelection("Selection removed from the batch list.");
    renderBatchBoxes();
    renderRegions();
    updateAnnotationSaveAvailability();
    return;
  }
  if (state.selectedTrack && state.tracks.includes(state.selectedTrack)) {
    state.tracks = state.tracks.filter((track) => track !== state.selectedTrack);
    clearCurrentSelection("Provisional detection removed from this analysis.");
    renderTracks();
    return;
  }
  if (state.selection) clearCurrentSelection("Selection cleared; no saved region was deleted.");
}

document.addEventListener("keydown", event => {
  const key = event.key.toLowerCase();
  const editing = event.target instanceof Element
    && Boolean(event.target.closest("input, textarea, select, [contenteditable='true']"));
  if (event.key === "Escape") {
    if (document.querySelector("dialog[open]")) return;
    if (editing) return;
    let cancelledMode = false;
    if (state.resizeDrag) {
      state.selection = {...state.resizeDrag.originalSelection};
      state.resizeDrag = null;
      drawOverlays();
      return;
    }
    if (state.moveDrag) {
      state.selection = {...state.moveDrag.originalSelection};
      state.moveDrag = null;
      drawOverlays();
      return;
    }
    if (state.adjustingCandidate) {
      state.adjustingCandidate = null;
      el("cv-status").textContent = "Candidate redraw cancelled; the original bounds remain.";
      cancelledMode = true;
    }
    if (state.drawingSignalBox) {
      cancelSingleBoxDrawing();
      cancelledMode = true;
    }
    if (state.batchBoxMode) {
      state.batchBoxMode = false;
      updateSingleBoxButton();
      renderBatchBoxes();
      drawOverlays();
      return;
    }
    if (state.adjustingGeometry || state.geometryDraft) {
      el("cancel-region-bounds").click();
      cancelledMode = true;
    }
    if (state.splitMode) {
      el("cancel-split").click();
      cancelledMode = true;
    }
    if (cancelledMode) return;
    if (state.selection) {
      clearCurrentSelection("Selection cleared.");
      return;
    }
    el("inspector").classList.add("hidden");
    return;
  }
  if ((event.ctrlKey || event.metaKey) && key === "z" && !editing && state.batchBoxUndo.length) {
    event.preventDefault();
    undoBatchBoxChange();
    return;
  }
  if (event.key === "Tab" && !editing && !event.ctrlKey && !event.metaKey && !event.altKey
    && state.inspectorMode === "annotate" && state.batchBoxes.length > 1) {
    const currentIndex = state.batchBoxes.indexOf(state.batchReviewBox);
    const direction = event.shiftKey ? -1 : 1;
    const nextIndex = currentIndex < 0
      ? (direction > 0 ? 0 : state.batchBoxes.length - 1)
      : (currentIndex + direction + state.batchBoxes.length) % state.batchBoxes.length;
    event.preventDefault();
    reviewBatchBox(nextIndex);
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !editing) {
    if (!el("save-annotations").disabled) {
      event.preventDefault();
      el("save-annotations").click();
    }
    return;
  }
  if (editing || event.ctrlKey || event.metaKey || event.altKey || event.repeat || document.querySelector("dialog[open]")) return;
  if (key === "b") {
    if (!state.waterfall || state.liveView || state.coverageOnly) return;
    event.preventDefault();
    el("batch-box-toggle").click();
    openInspector("label");
  } else if (event.key === "Delete" || event.key === "Backspace") {
    if (!state.selection && !state.editRegion && !state.selectedTrack && !state.batchReviewBox) return;
    event.preventDefault();
    deleteCurrentSelection();
  } else if (key === "m") {
    const mergeButton = [el("merge-selection-boxes"), el("merge-redrawn-regions")]
      .find((button) => button && !button.hidden && !button.disabled);
    if (!mergeButton) return;
    event.preventDefault();
    mergeButton.click();
  }
});
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
new ResizeObserver(() => { if (state.waterfall) { renderFrequencyNotches(); renderSignalLabels(); } }).observe(overlayCanvas.parentElement);

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
  if (!state.cvCoverageQueue) el("cv-coverage-queue-label-status").hidden = true;
  state.selectedTrack = null;
  state.batchReviewBox = null;
  el("cv-presence-row").hidden = true;
  el("cv-presence-confirmed").checked = false;
  state.editRegion = null;
  state.selectionMergeRegions = [];
  updateSelectionMergeControls();
  if (state.capture) renderRegionShortcuts();
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
  el("coverage-mask-status").textContent = "Start a window review to mark bounded subregions.";
  renderSceneCoverageMasks();
  updateCoverageMaskControls();
  setRegionSaveDisabled(!state.selection || state.liveView || !state.capture);
  syncLiveAnnotationSaveButton();
  el("save-region").textContent = "Save region label";
  setModeProposalSource(null);
  state.labelModeId = null;
  state.modeSearchRequest += 1;
  el("batch-unknown-row").hidden = true;
  el("batch-unknown-confirmed").checked = false;
  el("mode-search").value = "";
  el("mode-results").replaceChildren();
  for (const field of ["family", "method", "protocol", "variant", "notes"]) el(field).value = "";
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
  updateAnnotationSaveAvailability();
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
      const removedSources=new Set(operation.removed_source_region_ids||[]);
      const removedResults=new Set(operation.removed_result_region_ids||[]);
      const sources=operation.source_region_ids.map((id)=>`${id===region.id?"this region":id.slice(0,8)}${removedSources.has(id)?" (removed; history kept)":""}`).join(", ");
      const results=operation.result_region_ids.map((id)=>`${id===region.id?"this region":id.slice(0,8)}${removedResults.has(id)?" (removed; history kept)":""}`).join(", ");
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
  const list = el("signal-region-shortcuts");
  list.replaceChildren();
  const visibleRegions = state.regions.filter((region) => state.showSupersededRegions || region.lineage_state !== "superseded");
  el("signal-saved-count").textContent = String(state.regions.filter((region) => region.lineage_state !== "superseded").length);
  for (const region of visibleRegions) {
    const card = document.createElement("div");
    card.className = "saved-signal-card";
    if (state.editRegion?.id === region.id) card.classList.add("is-selected");
    const title = document.createElement("strong");
    const pendingEdit = state.windowRegionEdits.get(region.id);
    title.textContent = `${region.signal_id ? `${region.signal_id} · ` : ""}${[pendingEdit?.labelDraft.family, pendingEdit?.labelDraft.method, pendingEdit?.labelDraft.variant].filter(Boolean).join(" · ")
      || pendingEdit?.labelDraft.protocol || [region.family, region.method, region.variant].filter(Boolean).join(" · ")
      || region.protocol || regionIdentityCaption(region, pendingEdit?.labelDraft || region, pendingEdit?.labelDraft)}`;
    const detail = document.createElement("span");
    const rate = state.capture.sample_rate;
    const band = Number.isFinite(region.frequency_low_hz) && Number.isFinite(region.frequency_high_hz)
      ? `${region.frequency_low_hz.toFixed(0)}–${region.frequency_high_hz.toFixed(0)} Hz` : "band unknown";
    const review = region.lineage_state === "superseded" ? "superseded"
    : region.source === "prediction" ? "classifier suggestion · pending review"
      : region.source === "unknown" && region.review_state === "pending" ? "unknown · pending review"
      : region.review_state;
    const evidenceSources = [];
    if (region.evidence?.detector_proposal) evidenceSources.push("CV");
    if (region.evidence?.classifier_proposal) evidenceSources.push("classifier");
    const sourceSummary = evidenceSources.length ? `${evidenceSources.join(" + ")} · ` : "";
    detail.textContent = `${sourceSummary}${(region.sample_start / rate).toFixed(2)}–${((region.sample_start + region.sample_count) / rate).toFixed(2)} s · ${band} · ${review}`;
    const mergeSelect = document.createElement("label");
    mergeSelect.className = "saved-signal-merge";
    const mergeCheckbox = document.createElement("input");
    mergeCheckbox.type = "checkbox";
    mergeCheckbox.dataset.regionMergeId = region.id;
    mergeCheckbox.checked = state.mergeRegionIds.has(region.id);
    mergeCheckbox.disabled = state.liveView || region.lineage_state === "superseded";
    mergeCheckbox.setAttribute("aria-label", `Select ${title.textContent} from ${evidenceSources.join(" and ") || "saved annotations"} at ${(region.sample_start / rate).toFixed(2)} seconds for merge`);
    mergeCheckbox.addEventListener("change", () => {
      if (mergeCheckbox.checked) state.mergeRegionIds.add(region.id);
      else state.mergeRegionIds.delete(region.id);
      updateMergeControls();
    });
    mergeSelect.append(mergeCheckbox, document.createTextNode("Merge this piece"));
    const identitySelect = document.createElement("label");
    identitySelect.className = "saved-signal-identity";
    const identityCheckbox = document.createElement("input");
    identityCheckbox.type = "checkbox";
    identityCheckbox.checked = state.identityLinkSelection.has(region.id);
    identityCheckbox.disabled = state.liveView || region.lineage_state === "superseded";
    identityCheckbox.setAttribute("aria-label", `Select ${title.textContent} for same signal identity linking`);
    identityCheckbox.addEventListener("change", () => {
      if (identityCheckbox.checked) state.identityLinkSelection.add(region.id);
      else state.identityLinkSelection.delete(region.id);
      refreshIdentityLinkControls();
    });
    identitySelect.append(identityCheckbox, document.createTextNode("Same signal"));
    const actions = document.createElement("div");
    actions.className = "saved-signal-actions";
    const reviewButton = document.createElement("button");
    reviewButton.type = "button";
    reviewButton.textContent = "Edit identity";
    reviewButton.title = "Open the mode, modulation, and notes editor.";
    reviewButton.addEventListener("click", (event) => {
      selectSavedRegion(region);
    });
    const manage = document.createElement("button");
    manage.type = "button";
    manage.textContent = "Bounds";
    manage.title = "Open this signal's bounds, merge, delete, and history controls.";
    manage.addEventListener("click", () => selectSavedRegion(region, "signals"));
    actions.append(reviewButton, manage);
    card.append(title, detail, identitySelect, mergeSelect, actions);
    list.append(card);
  }
  updateMergeControls();
  if (!state.regions.length) list.textContent = "No saved signals yet. Draw a box or review a provisional candidate below.";
}
