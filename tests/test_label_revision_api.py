from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import httpx

from sigid.api import create_app
from sigid.dataset import CaptureStore, LabelRevisionConflict, SignalLabel
from sigid.synthetic import generate_bpsk_wav


def test_revision_api_preserves_original_history_bounds_and_rejects_stale_reviews(tmp_path):
    wav=tmp_path/'source.wav'
    generate_bpsk_wav(wav,'00110110'*32)
    root=tmp_path/'data'
    app=create_app(root)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            capture=(await client.post('/api/captures',files={'file':('test.wav',wav.read_bytes(),'audio/wav')})).json()
            base=f"/api/captures/{capture['id']}/regions"
            region=(await client.post(base,json={'sample_start':100,'sample_count':2000,'notes':'Uncertain observation'})).json()
            path=f"{base}/{region['id']}"
            revision=await client.post(path+'/labels',json={'expected_revision':1,'family':'PSK','notes':'Reviewed as PSK','sample_start':0})
            assert revision.status_code==201,revision.text
            updated=revision.json()
            assert updated['id']==region['id'] and updated['revision']==2
            assert updated['sample_start']==100 and updated['sample_count']==2000
            assert updated['source']=='human' and updated['review_state']=='accepted'
            assert len((await client.get(base)).json())==1
            stale=await client.post(path+'/labels',json={'expected_revision':1,'family':'FSK'})
            assert stale.status_code==409
            assert (await client.get('/api/captures/wrong/regions/'+region['id']+'/history')).status_code==404
            history=(await client.get(path+'/history')).json()
            assert [row['revision'] for row in history]==[1,2]
            assert history[0]['source']=='unknown' and history[0]['notes']=='Uncertain observation'
            assert history[1]['evidence']=={'reviewed_from_revision':1}
            # The same persisted history is returned by a fresh application instance.
            restarted=create_app(root)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restarted),base_url='http://test') as other:
                assert (await other.get(path+'/history')).json()==history
    asyncio.run(run())


def test_simultaneous_label_reviews_require_the_revision_each_operator_opened(tmp_path):
    wav=tmp_path/'source.wav'
    generate_bpsk_wav(wav,'00110110'*16)
    store=CaptureStore(tmp_path/'dataset.sqlite3')
    capture=store.add_wav_capture(wav,source_kind='wav')
    region=store.add_region(capture,sample_start=0,sample_count=1000)
    def revise(family):
        try:
            return store.revise_label(region,SignalLabel(family=family,source='human'),expected_revision=1)
        except LabelRevisionConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as workers:
        results=list(workers.map(revise,['PSK','FSK']))
    assert results.count(2)==1 and results.count('conflict')==1
    assert len(store.label_history(region))==2
