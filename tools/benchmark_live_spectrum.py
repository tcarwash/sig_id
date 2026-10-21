"""Measure encoding cost for 200 ms display increments from ten-second audio rings."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import struct
import time

import numpy as np

from sigid.live_spectrum import spectrum_packet


def benchmark():
    results=[]
    for rate in (48000,96000,192000):
        count=rate*10
        pcm=np.rint(16000*np.sin(2*np.pi*1000*np.arange(count)/rate)).astype('<i2')
        snapshot={'session_id':'benchmark','pcm':pcm.tobytes(),'channels':1,'sample_rate':rate,'sample_count':count,'sample_offset':0}
        after=((count-int(rate*0.2)-4096)//512)*512
        durations=[]
        for _ in range(50):
            start=time.perf_counter()
            packet=spectrum_packet(snapshot,after_sample=after,session_id='benchmark')
            durations.append((time.perf_counter()-start)*1000)
        size=struct.unpack('<I',packet[:4])[0]
        meta=json.loads(packet[4:4+size])
        results.append({'sample_rate':rate,'increment_seconds':0.2,'packet_bytes':len(packet),
                        'rows':meta['shape'][0],'frequency_bins':meta['shape'][1],
                        'encode_ms_median':float(np.median(durations)),
                        'encode_ms_p95':float(np.quantile(durations,0.95))})
    return {'schema_version':1,'host':platform.node(),'platform':platform.platform(),
            'scope':'in-process synthetic encoding benchmark only; excludes device, network and browser rendering',
            'iterations_per_rate':50,'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=benchmark()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
