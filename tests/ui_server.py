"""Isolated browser-test server with a paced synthetic audio device; no radio access."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Event, Thread
import time

import numpy as np

from sigid.api import create_app as workbench_app


class SyntheticAudio:
    CallbackStop = type('CallbackStop',(Exception,),{})
    CallbackAbort = type('CallbackAbort',(Exception,),{})

    def query_devices(self, device=None, kind=None):
        item={'name':'Synthetic UI input','max_input_channels':1,'default_samplerate':48000}
        return item if kind=='input' or device is not None else [item]

    def InputStream(self, **kwargs):
        return SyntheticStream(kwargs)


class SyntheticStream:
    def __init__(self, settings):
        self.settings=settings
        self.done=Event()

    def start(self):
        self.thread=Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        rate=self.settings['samplerate']
        count=rate//20
        start=0
        while not self.done.is_set():
            positions=np.arange(start,start+count)
            frequencies=np.where((positions//(rate//10))%2,1200,1000)
            samples=np.rint(16000*np.sin(2*np.pi*frequencies*positions/rate)).astype('<i2').reshape(-1,1)
            try:
                self.settings['callback'](samples,count,None,None)
            except (SyntheticAudio.CallbackAbort,SyntheticAudio.CallbackStop):
                break
            start+=count
            self.done.wait(0.05)

    def stop(self):
        self.done.set()
        self.thread.join(timeout=2)

    def close(self):
        self.stop()


def create_app():
    return workbench_app(Path(os.environ.get('SIG_ID_DATA_DIR','/tmp/sigid-ui-test-data')),audio_backend=SyntheticAudio())
