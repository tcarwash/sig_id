/** Investigate workspace navigation and exact decoder-attempt comparison. */
import assert from "node:assert/strict";
import {mkdtemp} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
const {chromium}=await import(process.env.SIGID_PLAYWRIGHT_MODULE || "playwright");
const base=process.env.SIGID_UI_URL || "http://127.0.0.1:8022";
const artifacts=await mkdtemp(join(tmpdir(),"sigid-investigate-workbench-"));
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:900}});
const errors=[];
page.on("pageerror",error=>errors.push(error.message));
await page.route("**/app.js",async route=>{
  const response=await route.fetch();
  await route.fulfill({response,body:`${await response.text()}\nglobalThis.__investigateTest={state,renderDecodeRuns,openInspector};`});
});
await page.route("**/api/fldigi/**",route=>route.fulfill({json:{connected:false,modes:[],text:"",end:0}}));
await page.route("**/api/decoder-backends/**",route=>route.fulfill({json:[]}));
try{
  await page.goto(base);
  await page.waitForFunction(()=>Boolean(globalThis.__investigateTest));
  await page.locator("#show-signals").click();
  const tabs=page.locator("#inspector-tabs");
  assert.equal(await tabs.getByRole("button",{name:"Investigate",exact:true}).count(),1,"decoder and raw analysis share one top-level activity");
  assert.equal(await tabs.locator('[data-inspector-tab="decode"]').count(),0);
  assert.equal(await tabs.locator('[data-inspector-tab="raw"]').count(),0);
  await tabs.getByRole("button",{name:"Investigate",exact:true}).click();
  await page.locator("#inspector").evaluate(node=>{node.dataset.detailLevel="simple";});
  const selection={sample_start:1200,sample_count:6000,frequency_low_hz:900,frequency_high_hz:1500};
  const runs=[
    {id:"attempt-a",status:"completed",current_mode:"RTTY",attempted_modes:["RTTY"],sample_start:1200,sample_count:6000,sample_rate:12000,carrier_hz:1100,backend:"fldigi-test",output_scope:"isolated_region_replay",association:"exact_capture_window",note:"Earlier attempt",output_text:"FIRST RESULT",detector_evidence:{proposal_id:"proposal-a"}},
    {id:"attempt-b",status:"completed",current_mode:"BPSK31",attempted_modes:["BPSK31"],sample_start:1200,sample_count:6000,sample_rate:12000,carrier_hz:1250,backend:"fldigi-test",output_scope:"isolated_region_replay",association:"exact_capture_window",note:"Second attempt",output_text:"SECOND RESULT"},
    {id:"attempt-c",status:"failed",current_mode:"QPSK31",attempted_modes:["QPSK31"],sample_start:1200,sample_count:6000,sample_rate:12000,carrier_hz:1400,backend:"fldigi-test",output_scope:"isolated_region_replay",association:"exact_capture_window",note:"No result",output_text:""},
  ];
  await page.evaluate(({selection,runs})=>{
    const {state,renderDecodeRuns}=globalThis.__investigateTest;
    Object.assign(state,{selection,capture:{id:"capture-test",sample_rate:12000,sample_count:24000},liveView:false,decodeRuns:runs,regions:[],inspectorMode:"classify",inspectorTab:"decode"});
    renderDecodeRuns();
    document.getElementById("inspector").classList.remove("hidden");
    globalThis.__investigateTest.openInspector("decode");
  },{selection,runs});
  assert.equal(await page.locator("#fldigi-parameters summary").textContent(),"Parameters","advanced controls have local disclosure in Simple view");
  assert.equal(await page.locator("#fldigi-carrier-hz").isVisible(),false,"parameters default collapsed");
  await page.locator("#fldigi-parameters summary").click();
  assert.equal(await page.locator("#fldigi-carrier-hz").isVisible(),true,"local Parameters disclosure works without global Advanced mode");
  await page.locator('[data-investigation-tool="raw"]').click();
  assert.equal(await page.locator('[data-investigation-pane="raw"]').isVisible(),true);
  assert.equal(await page.locator('[data-investigation-pane="decode"]').isVisible(),false);
  assert.deepEqual(await page.evaluate(()=>globalThis.__investigateTest.state.selection),selection,"tool switching keeps exact waterfall selection");
  await page.locator('[data-investigation-tool="decoder"]').click();
  assert.equal(await page.locator('[data-investigation-pane="decode"]').isVisible(),true);
  assert.equal(await page.locator('[data-investigation-pane="raw"]').isVisible(),false);
  const compare=page.locator("#compare-decode-runs");
  assert.equal(await compare.isDisabled(),true,"comparison requires two selected attempts");
  await page.locator('[data-compare-run-id="attempt-a"]').check();
  await page.locator('[data-compare-run-id="attempt-b"]').check();
  assert.equal(await compare.isEnabled(),true);
  await compare.click();
  const comparison=page.locator("#decode-run-comparison");
  assert.equal(await comparison.isVisible(),true);
  assert.equal(await comparison.locator(".decode-attempt-card").count(),2);
  assert.match(await comparison.textContent(),/FIRST RESULT/);
  assert.match(await comparison.textContent(),/SECOND RESULT/);
  assert.match(await comparison.textContent(),/1100 Hz/);
  assert.match(await comparison.textContent(),/1250 Hz/);
  assert.match(await comparison.textContent(),/1200–7200 samples/);
  await page.screenshot({path:join(artifacts,"investigate-desktop.png"),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.locator("[data-investigation-tool=decoder]").isVisible(),true);
  assert.equal(await page.locator("#decode-run-comparison").isVisible(),true);
  await page.screenshot({path:join(artifacts,"investigate-mobile.png"),fullPage:true});
  await page.setViewportSize({width:1440,height:900});
  await page.locator("#clear-decode-run-comparison").click();
  assert.equal(await comparison.isHidden(),true);
  assert.equal(await page.locator('[data-compare-run-id="attempt-a"]').isChecked(),false);
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:"passed",message:"unified Investigate tab, local parameters, source-preserving tool switches, exact attempt comparison",artifacts}));
}finally{await browser.close();}
