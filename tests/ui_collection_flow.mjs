/** End-to-end collection over real APIs, synthetic source and persisted run. */
import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {mkdtemp} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join, resolve} from "node:path";
const {chromium} = await import(process.env.SIGID_PLAYWRIGHT_MODULE || "playwright");
const base = process.env.SIGID_UI_URL || "http://127.0.0.1:8019";
const data = process.env.SIGID_UI_DATA_DIR;
assert.ok(data && resolve(data).startsWith(`${resolve(tmpdir())}/`), "requires explicit isolated temporary server data directory");
const root = await mkdtemp(join(tmpdir(), "sigid-collection-"));
const ids = JSON.parse(execFileSync(process.env.SIGID_PYTHON || ".venv/bin/python", ["-c", `
import json, wave
from pathlib import Path
from sigid.dataset import CaptureStore
p=Path(${JSON.stringify(root)})/'source.wav'
with wave.open(str(p),'wb') as w:
 w.setparams((1,2,12000,0,'NONE','not compressed')); w.writeframes(b'\\x01\\x00'*120000)
s=CaptureStore(Path(${JSON.stringify(data)})/'dataset.sqlite3')
c=s.add_wav_capture(p)
r=s.create_decoder_run(session_id='browser-collection',capture_id=c,region_id=None,track_id='track-example',sample_start=12000,sample_count=48000,sample_rate=12000,modulation_hypothesis='PSK',backend='fldigi-isolated-replay',modes=['BPSK31'],attempted_modes=['BPSK31'],current_mode='BPSK31',carrier_hz=1500,output_offset=0,status='completed',note='Synthetic persisted browser fixture',output_scope='isolated_region_replay',association='exact_capture_window',detector_evidence={'frequency_low_hz':1400,'frequency_high_hz':1600})
s.update_decoder_run(r,{'status':'completed','current_mode':'BPSK31','text':'RETAIN THIS EXACT ATTEMPT','note':'Synthetic persisted browser fixture'})
print(json.dumps({'capture':c,'run':r}))
`], {encoding: "utf8"}));
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
const errors = [];
page.on("pageerror", error => errors.push(error.message));
await page.route("**/api/fldigi/**", route => route.fulfill({json:{connected:false}}));
try {
  const devices = await (await page.request.get(`${base}/api/audio/devices`)).json();
  assert.ok(devices.every(d => d.name === "Synthetic UI input"));
  await page.goto(base);
  await page.locator("#show-library").click();
  await page.locator(`#capture-list [data-capture-id="${ids.capture}"] button`).click();
  await page.locator("#library-dialog").waitFor({state:"hidden"});
  await page.locator("#show-signals").click();
  await page.locator('[data-inspector-mode="classify"]').click();
  await page.locator('[data-inspector-tab="investigate"]').click();
  await page.locator('[data-investigation-tool="decoder"]').click();
  const collect = page.locator(".decode-run-row").filter({hasText:"BPSK31"}).getByRole("button",{name:"Collect example",exact:true});
  await collect.click();
  await page.waitForFunction(() => document.getElementById("collection-status").textContent.includes("pending"));
  assert.equal(await page.locator("#review-queue-dialog").isVisible(),false);
  const queue = await (await page.request.get(`${base}/api/live/review-queue`)).json();
  const candidate = queue.pending.find(c => c.source_metadata.collected_decoder_run_id === ids.run);
  assert.ok(candidate);
  assert.equal((await (await page.request.get(`${base}/api/captures/${candidate.id}/coverage`)).json()).length,0);
  await page.locator("#show-review-queue").click();
  await page.locator(".review-queue-row").filter({hasText:candidate.id.slice(0,8)}).getByRole("button",{name:"Review",exact:true}).click();
  await page.locator("#inspector").waitFor({state:"visible"});
  const saved = await (await page.request.get(`${base}/api/captures/${candidate.id}/regions`)).json();
  assert.equal(saved[0].review_state,"pending");
  await page.locator(".saved-signals-panel > summary").click();
  await page.locator("#signal-region-shortcuts").getByRole("button",{name:"Edit identity",exact:true}).first().click();
  await page.locator("#collected-attempt").waitFor({state:"visible"});
  assert.match(await page.locator("#collected-attempt").textContent(),/RETAIN THIS EXACT ATTEMPT/);
  await page.getByRole("button",{name:"Use suggested mode",exact:true}).click();
  await page.waitForFunction(() => document.getElementById("variant").value === "BPSK31");
  const unchanged = await (await page.request.get(`${base}/api/captures/${candidate.id}/regions`)).json();
  assert.equal(unchanged[0].review_state,"pending","using suggestion only edits the review draft");
  assert.equal(unchanged[0].mode_id,null);
  await page.locator("#save-annotations").click();
  await page.waitForFunction(() => document.getElementById("label-state").textContent.toLowerCase().includes("saved"));
  const reviewed = await (await page.request.get(`${base}/api/captures/${candidate.id}/regions`)).json();
  assert.equal(reviewed[0].variant,"BPSK31");
  assert.equal(await page.locator('#variant').inputValue(),'BPSK31','Save progress retains selected saved signal context');
  assert.equal(await page.locator('#collected-attempt').isVisible(),true,'collected evidence remains visible after saving');
  assert.equal((await (await page.request.get(`${base}/api/captures/${candidate.id}/coverage`)).json()).length,0);
  await page.screenshot({path:`${root}/collection-review-desktop.png`,fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.ok((await page.locator('.inspector-content').boundingBox()).height >= 250,'mobile review leaves usable space above persistent actions');
  await page.screenshot({path:`${root}/collection-review-mobile.png`,fullPage:true});
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:"passed",capture:candidate.id,artifacts:root}));
} finally {await browser.close();}
