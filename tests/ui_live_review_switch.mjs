/** Review a collected clip over an active synthetic live session and restore its selection. */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {mkdtemp} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";

const {chromium} = await import(process.env.SIGID_PLAYWRIGHT_MODULE || "playwright");
const base = process.env.SIGID_UI_URL || "http://127.0.0.1:8019";
const data = process.env.SIGID_UI_DATA_DIR;
assert.ok(data && resolve(data).startsWith(`${resolve(tmpdir())}/`), "requires an isolated temporary server data directory");
const root = await mkdtemp(join(tmpdir(), "sigid-live-review-switch-"));
const seeded = JSON.parse(execFileSync(process.env.SIGID_PYTHON || ".venv/bin/python", ["-c", `
import json, math, struct, wave
from pathlib import Path
from sigid.dataset import CaptureStore
root=Path(${JSON.stringify(root)})
p=root/'source.wav'
with wave.open(str(p),'wb') as w:
 w.setparams((1,2,12000,0,'NONE','not compressed'))
 w.writeframes(b''.join(struct.pack('<h',int(8000*math.sin(i*2*math.pi*1100/12000))) for i in range(120000)))
s=CaptureStore(Path(${JSON.stringify(resolve(data))})/'dataset.sqlite3')
c=s.add_wav_capture(p)
r=s.create_decoder_run(session_id='live-review-switch-fixture',capture_id=c,region_id=None,track_id='fixture-track',sample_start=12000,sample_count=48000,sample_rate=12000,modulation_hypothesis='PSK',backend='fldigi-isolated-replay',modes=['BPSK31'],attempted_modes=['BPSK31'],current_mode='BPSK31',carrier_hz=1100,output_offset=0,status='completed',note='Synthetic queued review fixture',output_scope='isolated_region_replay',association='exact_capture_window')
s.update_decoder_run(r,{'status':'completed','current_mode':'BPSK31','text':'SYNTHETIC REVIEW EVIDENCE','note':'Synthetic queued review fixture'})
print(json.dumps({'capture':c,'run':r}))
`], {encoding:"utf8"}));

const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
page.setDefaultTimeout(10000);
const errors = [];
page.on("pageerror", error => errors.push(error.message));
await page.route("**/app.js", async route => {
  const response = await route.fetch();
  await route.fulfill({response,body:`${await response.text()}\nglobalThis.__liveReviewSwitchTest = {state};`});
});
await page.route("**/api/fldigi/**", route => route.fulfill({json:{connected:false}}));
try {
  const devices = await (await page.request.get(`${base}/api/audio/devices`)).json();
  assert.ok(devices.length > 0 && devices.every(device => device.name === "Synthetic UI input"), "requires tests/ui_server.py synthetic input only");
  const collected = await page.request.post(`${base}/api/captures/${seeded.capture}/decode-runs/${seeded.run}/collect`, {data:{notes:"Synthetic live/review switch fixture"}});
  assert.equal(collected.status(),201,await collected.text());
  const queued = await collected.json();
  const capture = await (await page.request.get(`${base}/api/captures/${queued.capture_id}`)).json();

  await page.goto(base);
  await page.locator("#show-settings").click();
  const deviceSelect = page.locator("#audio-device");
  await page.waitForFunction(() => [...document.getElementById("audio-device").options].some(option => option.textContent.includes("Synthetic UI input")));
  const syntheticDevice = await deviceSelect.locator("option").evaluateAll(options => options.find(option => option.value !== "")?.value);
  assert.ok(syntheticDevice);
  await deviceSelect.selectOption(syntheticDevice);
  await page.locator('#settings-dialog [data-close-dialog]').click();
  await page.locator("#live-capture-mode").selectOption("scan_only");
  await page.locator("#live-start").click();
  await page.locator("#live-stop").waitFor({state:"visible"});
  await page.waitForFunction(() => document.getElementById("live-stop").disabled === false);
  await page.waitForFunction(() => document.getElementById("waterfall-state").textContent.includes("30 s display"),{timeout:20000});
  await page.waitForFunction(() => Number.isFinite(globalThis.__liveReviewSwitchTest.state.liveWindowStartSample)
    && Number.isFinite(globalThis.__liveReviewSwitchTest.state.liveWindowEndSample));
  // Use an exact interval from the active session; scan-only has no stable classifier track to click.
  await page.evaluate(() => {
    const {state} = globalThis.__liveReviewSwitchTest;
    const start = state.liveWindowStartSample;
    const count = Math.min(48000,state.liveWindowEndSample-start);
    state.selection = {sample_start:start,sample_count:count,frequency_low_hz:900,frequency_high_hz:1500};
    state.selectedTrack = {track_id:"synthetic-live-selection",status:"persistent"};
    document.getElementById("selection-summary").textContent = "Synthetic live selection";
    document.getElementById("selection-details").textContent = `${start}–${start+count} samples · 900–1500 Hz`;
    document.getElementById("inspector-context").textContent = "Live selection · synthetic review-switch fixture";
  });
  const before = await page.evaluate(() => ({
    selection:(({sample_start,sample_count,frequency_low_hz,frequency_high_hz})=>({sample_start,sample_count,frequency_low_hz,frequency_high_hz}))(globalThis.__liveReviewSwitchTest.state.selection),
    sessionId:globalThis.__liveReviewSwitchTest.state.liveSessionId,
    summary:document.getElementById("selection-summary").textContent,
    details:document.getElementById("selection-details").textContent,
    inspectorMode:globalThis.__liveReviewSwitchTest.state.inspectorMode,
    inspectorTab:globalThis.__liveReviewSwitchTest.state.inspectorTab,
  }));
  assert.ok(before.sessionId);
  assert.ok(Number.isInteger(before.selection.sample_start) && before.selection.sample_count > 0,"test starts from an exact live sample selection");
  const liveBefore = await (await page.request.get(`${base}/api/live`)).json();
  assert.equal(liveBefore.active,true);

  await page.locator("#show-review-queue").click();
  const row = page.locator(".review-queue-row").filter({hasText:queued.capture_id.slice(0,8)});
  const reviewButton = row.getByRole("button",{name:"Review",exact:true});
  assert.equal(await reviewButton.isEnabled(),true,"queued clips remain reviewable without stopping live acquisition");
  await reviewButton.click();
  await page.locator("#review-queue-dialog").waitFor({state:"hidden"});
  await page.waitForFunction(id => globalThis.__liveReviewSwitchTest.state.capture?.id === id,queued.capture_id);
  assert.equal(await page.locator("#capture-title").textContent(),capture.source_metadata.uploaded_filename || capture.source_metadata.device_name || capture.source_metadata.generator || "Audio recording");
  await page.locator("#live-review-banner").waitFor({state:"visible"});
  assert.match(await page.locator("#live-review-banner").textContent(),/monitoring continues/i);
  assert.equal(await page.locator("#live-stop").isEnabled(),true,"live stop control remains available while the clip is reviewed");
  assert.equal(await page.locator("#live-start").isEnabled(),false,"reviewing a clip cannot start a second capture");
  const liveDuring = await (await page.request.get(`${base}/api/live`)).json();
  assert.equal(liveDuring.active,true,"review view does not stop the live session");
  await page.waitForTimeout(1200);
  const liveAfter = await (await page.request.get(`${base}/api/live`)).json();
  assert.equal(liveAfter.active,true);
  assert.ok(liveAfter.frames_written > liveBefore.frames_written,"synthetic acquisition advances while reviewing the saved clip");
  assert.match(await page.locator("#live-status").textContent(),/Monitoring continues/i,"the live UI reports ongoing acquisition while the review canvas is active");
  await page.screenshot({path:join(root,"live-review-desktop.png"),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:join(root,"live-review-mobile.png"),fullPage:true});
  await page.setViewportSize({width:1440,height:1000});

  await page.locator("#return-to-live").click();
  await page.locator("#live-review-banner").waitFor({state:"hidden"});
  await page.waitForFunction(() => globalThis.__liveReviewSwitchTest.state.liveView === true);
  const after = await page.evaluate(() => ({
    selection:(({sample_start,sample_count,frequency_low_hz,frequency_high_hz})=>({sample_start,sample_count,frequency_low_hz,frequency_high_hz}))(globalThis.__liveReviewSwitchTest.state.selection),
    sessionId:globalThis.__liveReviewSwitchTest.state.liveSessionId,
    capture:globalThis.__liveReviewSwitchTest.state.capture,
    summary:document.getElementById("selection-summary").textContent,
    details:document.getElementById("selection-details").textContent,
    inspectorMode:globalThis.__liveReviewSwitchTest.state.inspectorMode,
    inspectorTab:globalThis.__liveReviewSwitchTest.state.inspectorTab,
  }));
  assert.deepEqual(after.selection,before.selection,"return restores the selected live sample/frequency bounds");
  assert.equal(after.sessionId,before.sessionId,"return restores the same live session");
  assert.equal(after.capture,null,"the live selection is not replaced by the reviewed capture");
  assert.equal(after.summary,before.summary,"selection summary is restored in the UI");
  assert.equal(after.details,before.details,"selection detail is restored in the UI");
  assert.equal(after.inspectorMode,before.inspectorMode,"investigation mode is restored");
  assert.equal(after.inspectorTab,before.inspectorTab,"inspector tab is restored");
  assert.equal(await page.locator("#save-annotations-complete").isHidden(),true,"review completion actions are cleared on return to live");
  assert.equal(await page.locator("#defer-review").isHidden(),true,"review deferral actions are cleared on return to live");
  assert.equal(await page.locator("#live-stop").isEnabled(),true);
  const liveReturned = await (await page.request.get(`${base}/api/live`)).json();
  assert.equal(liveReturned.active,true);
  assert.deepEqual(errors,[]);
  await page.screenshot({path:join(root,"live-return-selection.png"),fullPage:true});
  console.log(JSON.stringify({result:"passed",capture:queued.capture_id,session:before.sessionId,selection:after.selection,artifacts:root}));
} finally {
  if (await page.request.get(`${base}/api/live`).then(response => response.json()).then(status => status.active).catch(()=>false)) {
    await page.request.post(`${base}/api/live/stop`).catch(()=>{});
  }
  await browser.close();
}
