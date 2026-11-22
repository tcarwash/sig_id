# Signal Lab — repeatable multi-mode audio loopback

Signal Lab is a deterministic receive-only audio test passband for Signal ID. It combines SHA-256-pinned, independently synthesized decoder fixtures into a repeating noisy stream. Its catalog grows as modem-agent fixtures merge. It never operates radio hardware, tunes a rig, keys PTT, or sends RF. Samples are **synthetic**, not on-air validation evidence.

It outputs both a mono 16-bit WAV for direct Signal ID upload and a stereo WAV for PipeWire/PulseAudio playback, plus a JSON truth manifest recording mode, audio carrier, absolute sample offsets, expected text, and input fixture hashes for every burst.

## Starting a virtual loop

From a Signal ID checkout with audio dependencies installed, run:

    python -m pip install -e '.[audio,api,dev]'
    python -m sigid.signal_lab list
    python -m sigid.signal_lab start --modes rtty,bpsk31,ax25,cw --duration 100 --snr-db 22

The installed entrypoint **sig-id-signal-lab** runs the same commands.

In Signal ID, open **Settings → Audio input**, choose **sigid_signal_lab.monitor**, choose 48 kHz and 1 channel, and start live capture. This is a dedicated *null-sink monitor*, not a real radio input or physical speaker.

To check/stop playback:

    python -m sigid.signal_lab status
    python -m sigid.signal_lab stop

The player stays running after you close the browser. Start creates the null sink if absent and sends every audio sample explicitly to the named sink, never the default output. Stop terminates just the managed loop player. The sink remains available for later tests. Start refuses a second player using the same state directory. Log files and generated WAVs go to ~/.local/state/sig-id-signal-lab by default, or a supplied --state-dir. Requires the local pactl and paplay utilities and an accessible PipeWire/PulseAudio session.

## Choose modes, noise, and traffic patterns

The default **sparse** pattern puts each modem on a widely separated audio channel
and sends **one complete transmission at a time**, with only two bursts per mode
spread over the full cycle. For two stations transmitting simultaneously, use
--max-active 2. The scheduler intelligently pairs AX.25 with distant CW and
RTTY with BPSK31 when all four modes are selected; this minimizes cross-mode
interference rather than assuming each experimental decoder has a perfect filter.

Reliable quiet baseline: frequency-separated, **sparse** bursts, one transmitter at a time.

    python -m sigid.signal_lab start --modes cw,rtty,bpsk31,ax25 \
        --duration 100 --snr-db 23 --traffic sparse --max-active 1 --bursts-per-mode 2

Change the mode list or carrier positions by stopping the current player first:

    python -m sigid.signal_lab stop
    python -m sigid.signal_lab start --modes rtty,bpsk31 \
        --carriers rtty=740,bpsk31=2520 --duration 100 --snr-db 15

Two-station mode: maximum two simultaneous signals, with deliberate quiet intervals:

    python -m sigid.signal_lab stop
    python -m sigid.signal_lab start --modes cw,rtty,bpsk31,ax25 \
        --traffic sparse --max-active 2 --bursts-per-mode 2 --duration 100

Stress test: all stations transmit **simultaneously** in their own audio frequency lanes:

    python -m sigid.signal_lab stop
    python -m sigid.signal_lab start --modes rtty,bpsk31,ax25,cw --traffic parallel --snr-db 12

For deliberate spectral collisions, combine --traffic parallel with --allow-overlap and an explicit --carriers mapping. This can produce no-decodes or corrupted decodes on experimental front ends, which is meaningful stress evidence rather than a generator failure.

The default --traffic sparse provides a clean, less-crowded reference with distinct carrier frequencies, no continuous narrow QRM tones, at most --max-active 1 or 2 stations, and exactly --bursts-per-mode repeats per cycle. Their time slots are evenly spread rather than packed into the first few seconds. Seeded white noise remains present in gaps. Use --qrm to add the optional 735 and 2675 Hz interfering tones; --no-qrm to explicitly disable them. The older --traffic staggered cycles one transmitter at a time as often as the configured --gap allows, and --traffic parallel plays all separate stations simultaneously. The --gap setting affects staggered/parallel, not sparse, which allocates evenly spaced slots instead. More modes can be packed into the 160–2900 Hz passband with guard bands; if they cannot all fit simultaneously, the planner schedules groups in separate time windows.

### Longer transmissions for live detector/decoder tests

The conformance WAVs contain very short messages (AX.25 0.31 s, RTTY
1.47 s, BPSK31 1.54 s, CW 3.85 s). To test tracking across many seconds,
set `--burst-seconds` to a requested **minimum continuous station activity
window**. Signal Lab repeats **complete** pinned modem messages/frames, with
0.25-second pauses between messages. It does **not** slow down symbol timing,
truncate a packet, fabricate another payload or key a transmitter. Actual
transmission duration rounds *up* to the next full message boundary.

For four modes spread out and only one station active at a time, a good
starting point is 16-second transmissions in a 100-second cycle:

    python -m sigid.signal_lab stop
    python -m sigid.signal_lab start --modes cw,rtty,bpsk31,ax25 \
        --traffic sparse --max-active 1 --bursts-per-mode 1 \
        --duration 100 --burst-seconds 16 \
        --snr-db -5 --snr-reference-hz 2500

Each burst is listed once in the truth JSON, with `fixture_repetitions`,
`fixture_gap_seconds` and `embedded_messages` containing the **absolute
sample starts and lengths** of every real repeated frame. `expected_text`
contains the expected sequence (newline-separated), not a claim that the
experimental decoder can recover the whole sequence. A standalone
`embedded_messages` slice can be compared to the pinned original decoder
vector. `--burst-seconds 0` (default) preserves the original short fixture.

The sparse scheduler rejects combinations that cannot fit intact in the
configured time slots. For 4 modes and `--burst-seconds 16`, choose
`--bursts-per-mode 1` or a longer cycle. The maximum burst request is
60 seconds and the overall loop length still caps at 300 seconds.

### Default carrier plan

| Mode | Audio carrier | Expected native decode |
| --- | --- | --- |
| CW | 260 Hz | CQ TEST |
| RTTY 45.45 baud ITA2 | 740 Hz | CQ TEST |
| BPSK31 | 2520 Hz | cq de |
| AX.25 Bell 202 AFSK | mark 1200 Hz, space 2200 Hz | HELLO WORLD, valid FCS |

AX.25's current first-party decoder expects **fixed 1200/2200 Hz tones**, regardless of the selected window's carrier-hz hint. Signal Lab intentionally keeps AX.25 at those frequencies by default. A manually shifted AX.25 test signal is possible for stress testing but **not guaranteed to decode natively** until AX.25 retuning is implemented.

A mode is listed only when there is a **synthetic positive WAV fixture**, explicit source carrier, verified SHA-256, and expected text/frames. The general signal-label mode catalog is not a simulator catalog. Unmerged partial or raw-symbol-only decoders do not count. You can supply --fixtures-root to test another complete agent fixture directory; that does not imply the installed decoder supports it.

Unknown modulations have their approximate spectrum estimated from the fixture before frequency packing; this is placement guidance, not emissions characterization. Signal Lab frequency-shifts the analytic audio waveform, not the RF carrier. Exact bandwidth and feasibility remain modem-specific.

## Render files without running an audio device

    python -m sigid.signal_lab render --modes cw,rtty,bpsk31,ax25 \
        --duration 48 --seed 71062 --snr-db 20 \
        --output /tmp/sigid-multi-mode.wav

Files:

- /tmp/sigid-multi-mode.wav — 48-kHz mono PCM16; import directly into Signal ID
- /tmp/sigid-multi-mode-stereo.wav — identical audio in both channels; used by the loop player
- /tmp/sigid-multi-mode.json — sample-accurate truth metadata, fixture hashes and parameters

### Weak-signal testing below 0 dB SNR

Signal Lab accepts **--snr-db from -30 to +55 dB**, including fractional
negative values. The SNR compares the active modem's nominal RMS signal
power against *white noise integrated over the reference bandwidth*. In the
legacy/default configuration, the reference is the entire audio Nyquist
bandwidth (24 kHz for a 48-kHz recording), so a narrowband demodulator may
report a much better SNR after filtering.

For radio-style weak-signal tests, explicitly choose **2500 Hz** as the
reference bandwidth. A narrower noise reference adds enough full-band white
noise to achieve the requested in-band SNR; the generator does **not** merely
subtract decibels from the display level:

    # Moderate weak-signal example, one synthetic modem at a time:
    python -m sigid.signal_lab start --modes cw,rtty,bpsk31,ax25 \
        --traffic sparse --max-active 1 --duration 100 \
        --snr-db -5 --snr-reference-hz 2500

    # Extremely weak, likely to fail in the experimental native decoders:
    python -m sigid.signal_lab render --modes bpsk31 \
        --snr-db -20 --snr-reference-hz 2500 \
        --output /tmp/weak-bpsk31.wav

The generated JSON truth manifest records the requested SNR, reference
bandwidth, nominal RMS signal and noise levels **after the common amplitude
normalization gain**, so repeated tests can be compared. Noise and signal
are scaled together before PCM16 conversion: the output is not clipped to
make a low SNR fit. Very weak signals are expected to be visually submerged
and may produce native-decoder abstentions; the simulator's job is to
synthesize the signal, not claim it decoded successfully.

The default signal RMS is 0.17 per station before overlap. Default SNR 19 dB adds seeded white noise. The default CLI renders a 100-second sparse loop with two complete bursts per selected mode, without continuous interfering tones. The legacy staggered/parallel modes include weak interfering tones at 735 and 2675 Hz unless disabled; sparse mode defaults to noise only. The mix is linearly peak-normalized if needed; the truth manifest records its gain. Use --snr-db 25 for a clean control, --snr-db -5 with --snr-reference-hz 2500 for more realistic weak-signal tests, or down to -30 to explore the noise floor. Sample rates 12, 24 and 48 kHz are supported; use 48 kHz for the virtual audio input.

## Troubleshooting and safety

- **No monitor in device list:** start the simulator and refresh Signal ID's audio input list. A server launched outside your desktop session may need XDG_RUNTIME_DIR=/run/user/<uid> and PULSE_SERVER pointing to its local PulseAudio socket.
- **Live capture overrun:** some PortAudio inputs deliver 32-frame callbacks, overwhelming the Python queue. Fix PR #8 requests 1024-frame callbacks; this is a separate application concern.
- **Decoded CW/AX.25 fails under parallel traffic:** return to staggered with --snr-db 23 and compare. Existing native decoders are experimental and not full multichannel skimmers.
- **A mode is not listed:** run list. Only complete pinned audio fixtures count. The generator does not fabricate messages for unfinished modes.
- **A previous manual loop is still playing:** stop that manually launched loop before using this manager to avoid doubled audio.
- **No radio control:** the simulator uses explicit named null-sink playback only. For actual received signals, select the real input yourself and stop the simulator.

Focused verification:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_signal_lab.py
