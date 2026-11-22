import assert from "node:assert/strict";
import test from "node:test";
import {summarizeDecoderResult} from "../sigid/web/decoder-result.mjs";

test("binary candidate frames and soft decisions remain visible but unverified", () => {
  const result = {
    source:"sigid-independent-implementation", decoder_id:"packet_fixture",
    status:"partial", version:"fixture-v1", text:"", tokens:[],
    frames:[{
      start_sample:2100, end_sample:2500, byte_count:3, bit_length:20,
      payload_base64:"AQID", encoding:"base64", protocol_hint:"synthetic-packet",
      integrity:[{kind:"crc",algorithm:"CRC-16/X25",outcome:"passed",corrected_bits:null}],
    }],
    soft_bits:[{start_sample:2150,end_sample:2450,
      encoding:"llr_positive_is_one",bit_count:4,values:[-3,0,1,6]}],
    evidence:{}, note:"Synthetic test only",
  };
  const summary=summarizeDecoderResult(result);
  assert.match(summary,/Partial candidate/);
  assert.match(summary,/Candidate binary frames: 1/);
  assert.match(summary,/3 bytes, 20 bits · samples 2100–2500/);
  assert.match(summary,/crc CRC-16\/X25: passed/);
  assert.match(summary,/Soft decision evidence: 4 bits in 1 span/);
  assert.match(summary,/not a calibrated probability or confirmed identification/);
  assert.doesNotMatch(summary,/AQID/);
  assert.doesNotMatch(summary,/\[-3,0,1,6\]/);
});

test("abstained decoder cannot imply confirmed result even with candidate frames", () => {
  const summary = summarizeDecoderResult({
    source:"sigid-independent-implementation", decoder_id:"packet_fixture",
    status:"no_decode",frames:[{start_sample:1,end_sample:5,byte_count:2,bit_length:16,
      integrity:[{kind:"crc",algorithm:"test",outcome:"failed"}]}],
    soft_bits:[],evidence:{},tokens:[],text:"",note:"CRC failed",
  });
  assert.match(summary,/Abstained — no reliable decode/);
  assert.match(summary,/crc test: failed/);
  assert.match(summary,/CRC failed/);
});
