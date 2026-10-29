/** Saved annotation corner resizing: exact screen-space target and draft-mode behavior. */
import assert from "node:assert/strict";
const {chromium}=await import(process.env.SIGID_PLAYWRIGHT_MODULE || "playwright");
const base=process.env.SIGID_UI_URL || "http://127.0.0.1:8021";
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1440,height:900}});
const errors=[];
page.on("pageerror",error=>errors.push(error.message));
await page.addInitScript(()=>{
  window.__handleRects=[];
  const original=HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext=function(...args){
    const context=original.apply(this,args);
    if(this.id!=="overlay") return context;
    return new Proxy(context,{get(target,key){
      if(key==="fillRect") return (...rect)=>{if(target.fillStyle==="#72fff0")window.__handleRects.push(rect);return target.fillRect(...rect);};
      const value=Reflect.get(target,key,target);
      return typeof value==="function"?value.bind(target):value;
    },set(target,key,value){return Reflect.set(target,key,value,target);}});
  };
});
await page.route("**/app.js",async route=>{
  const response=await route.fetch();
  await route.fulfill({response,body:`${await response.text()}\nglobalThis.__resizeTest={state,overlayCanvas,drawOverlays,resizeHandleAt};`});
});
try{
  await page.goto(base);
  await page.waitForFunction(()=>Boolean(globalThis.__resizeTest));
  const result=await page.evaluate(()=>{
    const {state,overlayCanvas,drawOverlays,resizeHandleAt}=globalThis.__resizeTest;
    document.getElementById("empty-state").classList.add("hidden");
    document.getElementById("workspace").classList.remove("hidden");
    const bins=512,rows=1024,rate=12000;
    state.liveView=false;state.coverageOnly=false;state.adjustingGeometry=false;
    state.inspectorMode="annotate";state.batchBoxMode=true;state.batchReviewBox=null;
    state.editRegion={id:"saved-region"};state.selectedTrack=null;state.regions=[];state.tracks=[];state.cvDetections=[];state.batchBoxes=[];
    state.selection={sample_start:1000,sample_count:4000,frequency_low_hz:500,frequency_high_hz:1500,x1:100,y1:100,x2:300,y2:600};
    state.waterfall={sample_rate:rate,sample_count:120000,hop_size:120,fft_size:512,
      frequency_hz:Array.from({length:bins},(_,i)=>i*3000/(bins-1)),sample_starts:Array.from({length:rows},(_,i)=>i*100),
      db:Array.from({length:rows},()=>Array(bins).fill(-50))};
    state.view={frequencyLowHz:0,frequencyHighHz:3000,timeStartSample:0,timeEndSample:120000,followTime:false};
    overlayCanvas.width=bins;overlayCanvas.height=rows;overlayCanvas.style.width="700px";overlayCanvas.style.height="300px";
    drawOverlays();
    const box=overlayCanvas.getBoundingClientRect();
    const sx=overlayCanvas.width/box.width,sy=overlayCanvas.height/box.height;
    const cssDelta=8;
    const horizontal=resizeHandleAt({x:100+cssDelta*sx,y:100});
    const vertical=resizeHandleAt({x:100,y:100+cssDelta*sy});
    const cyan=window.__handleRects.filter(([, ,w,h])=>w>0&&h>0);
    return {horizontal:horizontal?.name||null,vertical:vertical?.name||null,sx,sy,box:{x:box.x,y:box.y,width:box.width,height:box.height},selection:{...state.selection},batchBoxMode:state.batchBoxMode,editRegion:state.editRegion,
      handles:cyan.slice(-4).map(([, ,w,h])=>({widthCss:w/sx,heightCss:h/sy}))};
  });
  assert.equal(result.horizontal,"top-left","an 8 CSS-pixel horizontal offset remains inside the grab target");
  assert.equal(result.vertical,"top-left","an 8 CSS-pixel vertical offset remains inside the grab target");
  assert.equal(result.handles.length,4,"all four visible saved-region handles are drawn");
  for(const handle of result.handles){
    assert.ok(Math.abs(handle.widthCss-handle.heightCss)<0.75,`handle should be square in CSS pixels: ${JSON.stringify(handle)}`);
    assert.ok(handle.widthCss>=8&&handle.widthCss<=12,`handle should have a usable, bounded target: ${JSON.stringify(handle)}`);
  }
  const corner={x:result.box.x+100/result.sx,y:result.box.y+100/result.sy};
  await page.mouse.move(corner.x+8,corner.y+8);
  await page.mouse.down();
  await page.mouse.move(corner.x+20,corner.y+20,{steps:3});
  const dragStarted=await page.evaluate(()=>({active:Boolean(globalThis.__resizeTest.state.resizeDrag),selection:{...globalThis.__resizeTest.state.selection}}));
  await page.mouse.up();
  assert.equal(dragStarted.active,true,"pointerdown on stretched canvas handle starts a resize, not a new draw");
  assert.notEqual(dragStarted.selection.x1,result.selection.x1,"pointer drag changes selected signal bounds");
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({result:"passed",handles:result.handles,dragStarted:dragStarted.active}));
}finally{await browser.close();}
