import assert from "node:assert/strict";
import test from "node:test";
import {pruneLiveCandidates} from "../sigid/web/live-candidate-history.mjs";

const mk=(i,start,count=1000)=>({
  track_id:`s:cv:${i}`, sample_start:start, sample_count:count,
  frequency_low_hz:500, frequency_high_hz:700,
});
test("recent, overlapping, and selected live tracks remain actionable",()=>{
  const list=[mk(1,0),mk(2,20_000),mk(3,90_000),mk(4,100_000)];
  const recent=pruneLiveCandidates(list,120_000,1000,{retainSeconds:75});
  assert.deepEqual(recent.map(x=>x.track_id),["s:cv:3","s:cv:4"]);
  const pinned=pruneLiveCandidates(list,120_000,1000,{retainSeconds:75,selectedTrackId:"s:cv:1"});
  assert.deepEqual(pinned.map(x=>x.track_id),["s:cv:1","s:cv:3","s:cv:4"]);
});
test("hard browser cap retains newest tracks and pinned selection",()=>{
  const many=Array.from({length:400},(_,i)=>mk(i,i*1000,300));
  const result=pruneLiveCandidates(many,400_000,1000,{maxItems:15,retainSeconds:500,selectedTrackId:"s:cv:1"});
  assert.equal(result.length,15);
  assert.equal(result[0].track_id,"s:cv:1");
  assert.equal(result.at(-1).track_id,"s:cv:399");
});
test("original evidence objects are not changed",()=>{
  const records=[mk(1,500),mk(2,600)];
  assert.deepEqual(pruneLiveCandidates(records,600,1000),records);
  assert.equal(records[0].track_id,"s:cv:1");
});
