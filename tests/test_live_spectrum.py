from __future__ import annotations

import asyncio
import json
import struct

import httpx
import numpy as np
import pytest

from sigid.api import create_app
from sigid.live_spectrum import spectrum_packet
from sigid.spectrum import waterfall_samples


def snapshot(total, offset=0, session='session-a', rate=8000):
    values=np.rint(np.sin(2*np.pi*1000*np.arange(offset,total)/rate)*12000).astype('<i2')
    return {'pcm':values.tobytes(),'sample_rate':rate,'channels':1,'sample_offset':offset,
            'sample_count':total,'window_sample_count':total-offset,'session_id':session,'active':True}


def unpack(packet):
    size=struct.unpack('<I',packet[:4])[0]
    metadata=json.loads(packet[4:4+size])
    values=np.frombuffer(packet[4+size:],dtype='<f4').reshape(metadata['shape'])
    return metadata,values


def test_incremental_packets_match_absolute_fft_grid_without_replaying_rows():
    first,rows=unpack(spectrum_packet(snapshot(5000),fft_size=256,hop_size=64))
    last=first['sample_starts'][-1]
    second,new_rows=unpack(spectrum_packet(snapshot(6000),after_sample=last,session_id='session-a',fft_size=256,hop_size=64))
    assert second['sample_starts'][0]==last+64
    assert set(first['sample_starts']).isdisjoint(second['sample_starts'])
    assert not second['gap_before']
    expected=waterfall_samples(np.frombuffer(snapshot(6000)['pcm'],dtype='<i2')/32768,8000,fft_size=256,hop_size=64,max_rows=1024)
    bins=len(second['frequency_hz'])
    expected_rows=np.asarray(expected['db'])[:, :bins]
    np.testing.assert_array_equal(np.concatenate((rows,new_rows)),expected_rows)
    assert max(second['frequency_hz'])<=3000
    assert len(spectrum_packet(snapshot(6000),after_sample=last,session_id='session-a',fft_size=256,hop_size=64)) < len(json.dumps(expected))


def test_ring_trim_gaps_and_session_change_preserve_sample_positions():
    item,rows=unpack(spectrum_packet(snapshot(12000,offset=5037),after_sample=256,session_id='session-a',fft_size=256,hop_size=64,max_rows=20))
    assert len(rows)==20
    assert item['gap_before']
    assert all(position%64==0 and position>=5037 for position in item['sample_starts'])
    fresh,_=unpack(spectrum_packet(snapshot(2000,session='new'),after_sample=100000,session_id='old',fft_size=256,hop_size=64))
    assert fresh['session_id']=='new'
    assert fresh['sample_starts'][0]==0
    assert not fresh['gap_before']
    empty,_=unpack(spectrum_packet(snapshot(128),fft_size=256,hop_size=64))
    assert empty['sample_starts']==[]


def test_live_spectrum_endpoint_and_analysis_without_redundant_image(tmp_path):
    app=create_app(tmp_path)
    app.state.live_capture.analysis_audio=lambda:snapshot(8000)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            response=await client.get('/api/live/spectrum',params={'fft_size':256,'hop_size':64})
            assert response.status_code==200
            assert response.headers['content-type']=='application/octet-stream'
            assert response.headers['cache-control']=='no-store'
            meta,rows=unpack(response.content)
            assert meta['session_id']=='session-a' and len(rows)>0
            analysis=await client.get('/api/live/analysis',params={'fft_size':256,'hop_size':64,'include_spectrum':'false'})
            assert analysis.status_code==200
            assert analysis.json()['spectrum'] is None
            assert analysis.json()['tracks']
            assert (await client.get('/api/live/spectrum?fft_size=300')).status_code==422
            app.state.live_capture.analysis_audio=lambda:None
            assert (await client.get('/api/live/spectrum')).status_code==409
    asyncio.run(run())
