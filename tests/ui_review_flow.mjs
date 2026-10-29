/** Isolated review UI regression: run against tests/ui_server.py, never operator data. */
import assert from 'node:assert/strict';
import { mkdtemp, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import {tmpdir} from 'node:os';
const { chromium } = await import(process.env.SIGID_PLAYWRIGHT_MODULE || 'playwright');
const root = await mkdtemp(join(tmpdir(), 'sigid-review-'));
execFileSync(process.env.SIGID_PYTHON || '.venv/bin/python', ['-c', `import wave, math, struct
with wave.open(${JSON.stringify(join(root, 'review.wav'))}, 'wb') as w:
 w.setparams((1,2,12000,0,'NONE','not compressed'))
 w.writeframes(b''.join(struct.pack('<h', int(12000*math.sin(i*2*math.pi*1000/12000))) for i in range(240000)))
`]);
const browser = await chromium.launch({headless:true});
const page = await browser.newPage({viewport:{width:1440,height:1000}});
page.setDefaultTimeout(6000);
const errors = [];
page.on('pageerror', e => errors.push(e.message));
const base = process.env.SIGID_UI_URL || 'http://127.0.0.1:8027';
const captures = [];
let batchPosts = 0, coveragePosts = 0;
await page.route('**/api/fldigi/**', route => route.fulfill({json:{connected:false}}));
await page.route('**/api/live/review-queue', async route => {
 const pending = [];
 for (const capture of captures) {
  const records = await (await page.request.get(`${base}/api/captures/${capture.id}/coverage`)).json();
  if (!records.some(r => r.state === 'complete' && !r.is_stale)) pending.push(capture);
 }
 await route.fulfill({json:{pending,pending_count:pending.length,reviewed_count:captures.length-pending.length}});
});
await page.route('**/regions/batch', async route => { batchPosts++; await route.continue(); });
await page.route('**/coverage', async route => { if(route.request().method()==='POST') coveragePosts++; await route.continue(); });
async function draw(x0=.20,x1=.38) {
 const b=await page.locator('#overlay').boundingBox();
 await page.mouse.move(b.x+b.width*x0,b.y+b.height*.25); await page.mouse.down();
 await page.mouse.move(b.x+b.width*x1,b.y+b.height*.6,{steps:5}); await page.mouse.up();
}
try {
 const devices = await (await page.request.get(`${base}/api/audio/devices`)).json();
 assert.ok(devices.every(d => d.name === 'Synthetic UI input'), 'requires isolated synthetic server');
 for(let i=0;i<3;i++) {
  const response=await page.request.post(`${base}/api/captures`,{multipart:{file:{name:`review-${i}-${Date.now()}.wav`,mimeType:'audio/wav',buffer:await readFile(join(root,'review.wav'))}}});
  assert.equal(response.status(),200);
  const capture=await response.json();
  capture.source_metadata={...capture.source_metadata,auto_review_queue:true,device_name:`Review ${i}`,window_start_sample:0};
  captures.push(capture);
 }
 await page.goto(base);
 await page.locator('#show-review-queue').click();
 await page.locator('#review-queue-list').getByRole('button',{name:'Review',exact:true}).first().click();
 await page.locator('#review-queue-dialog').waitFor({state:'hidden'});
 await page.locator('[data-inspector-mode="annotate"]').click();
 await page.locator('.selection-panel > .local-options-disclosure > summary').click();
 await draw();
 await page.locator('#batch-box-list').getByRole('button',{name:'Select to label',exact:true}).first().click();
 await page.locator('#family').fill('FSK');
 await page.locator('#notes').fill('Keep this draft while investigating');
 const selectionDetails=await page.locator('#selection-details').textContent();
 await page.locator('[data-inspector-mode="classify"]').click();
 await page.locator('[data-inspector-tab="investigate"]').click();
 await page.locator('[data-investigation-tool="raw"]').click();
 await page.locator('[data-inspector-mode="annotate"]').click();
 assert.equal(await page.locator('#notes').inputValue(),'Keep this draft while investigating','workflow switches preserve selected draft fields');
 assert.equal(await page.locator('#selection-details').textContent(),selectionDetails);
 assert.equal(await page.locator('#save-annotations').isVisible(),true,'Save progress must be visible in Annotate');
 assert.equal(await page.locator('#save-annotations').textContent(),'Save progress');
 await page.locator('#save-annotations').click();
 await page.waitForFunction(()=>document.getElementById('label-state').textContent.includes('saved'));
 assert.equal(await page.locator('#inspector').isVisible(),true);
 assert.equal(coveragePosts,0,'Save progress does not complete coverage');
 const saved=await (await page.request.get(`${base}/api/captures/${captures[0].id}/regions`)).json();
 assert.equal(saved.length,1);
 assert.equal(saved[0].family,'FSK');
 await draw(.55,.7);
 await page.locator('#batch-box-list').getByRole('button',{name:'Select to label',exact:true}).first().click();
 await page.locator('#batch-unknown-confirmed').check();
 let failCoverage=true;
 await page.route('**/coverage',async route=>{
  if(route.request().method()==='POST' && failCoverage) {
   failCoverage=false; coveragePosts++;
   await route.fulfill({status:503,json:{detail:'Injected coverage outage'}});
  } else await route.fallback();
 });
 await page.locator('#save-annotations-complete').click();
 await page.waitForFunction(()=>document.getElementById('label-state').textContent.includes('Injected coverage outage'));
 assert.match(await page.locator('#label-state').textContent(),/Annotations saved.*completion pending/);
 assert.equal(await page.locator('#save-annotations-complete').textContent(),'Retry completion');
 const postsAfterSave=batchPosts;
 await page.locator('#save-annotations-complete').click();
 await page.waitForFunction(name=>document.getElementById('capture-title').textContent===name,captures[1].source_metadata.uploaded_filename);
 assert.equal(batchPosts,postsAfterSave,'completion retry must not resave annotations');
 assert.equal(await page.locator('#review-queue-dialog').isVisible(),false);
 await page.locator('#inspector').waitFor({state:'visible'});
 const completeRecords=await (await page.request.get(`${base}/api/captures/${captures[0].id}/coverage`)).json();
 assert.equal(completeRecords.length,1);
 assert.equal(completeRecords[0].sample_start,0);
 assert.equal(completeRecords[0].sample_count,120000,'coverage stays in reviewed window');
 await draw(.2,.35);
 await page.locator('#defer-review').click();
 await page.locator('#annotation-draft-dialog').waitFor({state:'visible'});
 await page.locator('#annotation-draft-dialog button[value="stay"]').click();
 assert.equal(await page.locator('#capture-title').textContent(),captures[1].source_metadata.uploaded_filename);
 assert.equal((await (await page.request.get(`${base}/api/captures/${captures[1].id}/coverage`)).json()).length,0);
 await page.locator('#defer-review').click();
 await page.locator('#annotation-draft-dialog button[value="save"]').click();
 await page.waitForFunction(name=>document.getElementById('capture-title').textContent===name,captures[2].source_metadata.uploaded_filename);
 await page.locator('#inspector').waitFor({state:'visible'});
 assert.equal((await (await page.request.get(`${base}/api/captures/${captures[1].id}/regions`)).json()).length,1,'defer can persist unfinished work');
 assert.equal((await (await page.request.get(`${base}/api/captures/${captures[1].id}/coverage`)).json()).length,0,'defer never asserts completeness');
 await page.locator('.selection-panel > .local-options-disclosure > summary').click();
 assert.equal(await page.locator('#save-annotations-close').isVisible(),false,'optional save actions stay behind local label options');
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.locator('#save-annotations').isVisible(),true);
 assert.equal(await page.locator('#save-annotations-complete').isVisible(),true);
 assert.equal(await page.locator('#defer-review').isVisible(),true);
 await page.setViewportSize({width:1440,height:1000});
 const crossing=await page.request.post(`${base}/api/captures/${captures[2].id}/regions`,{data:{sample_start:100000,sample_count:40000,frequency_low_hz:900,frequency_high_hz:1300}});
 assert.equal(crossing.status(),201);
 await page.locator('#show-review-queue').click();
 await page.locator('.review-queue-row').filter({hasText:captures[2].id.slice(0,8)}).getByRole('button',{name:'Review',exact:true}).click();
 await page.locator('#inspector').waitFor({state:'visible'});
 await page.locator('#save-annotations-complete').click();
 await page.waitForFunction(()=>document.getElementById('label-state').textContent.includes('still needs a label'));
 assert.equal((await (await page.request.get(`${base}/api/captures/${captures[2].id}/coverage`)).json()).length,0,'crossing-window unknowns must be reviewed too');
 await page.screenshot({path:join(root,'review-desktop.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});
 await page.screenshot({path:join(root,'review-mobile.png'),fullPage:true});
 assert.deepEqual(errors,[]);
 console.log(JSON.stringify({result:'passed',batchPosts,coveragePosts,artifacts:root}));
} finally { await browser.close(); }
