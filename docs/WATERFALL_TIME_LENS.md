# Waterfall Time Lens — recent samples expanded, old samples retained

Signal ID's live waterfall can display 60 seconds of history using a **logarithmic-like time axis**. The newest few seconds get greater vertical resolution while older samples remain visible in compressed form. The radio, FFT, sound capture, classifier, detector and decoder all operate on the original absolute sample positions. This feature affects the **display only**.

Use the **Time lens** control beside **Debug CV bounds** below the waterfall:

| Setting | History | Visual behavior |
|---|---:|---|
| Off · 30 s linear | 30 s | Previous waterfall behavior |
| Gentle · 60 s | 60 s | Expands recent seconds moderately (8-second focal scale) |
| Strong · 60 s | 60 s | Gives the newest 1–5 seconds much more space (1.5-second focal scale) |

The preference persists locally in the browser. Time Lens only applies to the live view and its frozen snapshot; saved recording review remains linear. Switching modes preserves frequency zoom, labels and sample-accurate selections. A frozen snapshot can be switched between the three scales without advancing time.

The upper part of the display is **Now**, and ticks at −1, −5, −15 and −30 seconds indicate the nonlinear age mapping. The bottom of a lens view is −60 seconds. The mapping is continuous, monotone and invertible:

    age_fraction = (view_end_sample - absolute_sample) / displayed_sample_count
    y_fraction = log1p(age_fraction / focus_fraction) / log1p(1 / focus_fraction)
    focus_fraction = focus_seconds / visible_seconds

For the default linear view, the map is simply `y_fraction = age_fraction`. Canvas pointer-to-sample conversion uses the exact inverse `expm1` transform; drawing a box, frequency notches, candidate overlays, tooltips, hover and dragging all continue to reference the correct **original sample indices**. A signal's apparent height is not its physical duration once the lens is enabled: use the sample-based metadata instead.

## Visual fidelity and performance

The 60-second view reuses the existing bounded live FFT-row cache. It **does not request additional FFTs or extend the decoder audio analysis ring** (which remains 10 seconds). The canvas keeps the former 30-second output raster height even in 60-second mode. Each output pixel row is inverse-mapped onto its closest real FFT row rather than stretching the source in frequency; missing FFT data beyond the capture-gap threshold stays blank, not fabricated.

Colored FFT rows are cached for the duration of one draw to avoid repeatedly converting the same source row as the newest seconds expand. The history is resampled only in the vertical **display** dimension. Compression necessarily loses some older visual detail; it does not discard the original capture or alter saved signal coordinates.

In an isolated headless Chromium test on the development desktop, switching
between modes took approximately **15–45 ms per redraw** at 48 kHz and a
257 × 2813 canvas, with no JavaScript exceptions or missing modules. These
are indicative one-off timings, not guaranteed end-to-end UI framerates.

The time lens uses the same WebGL-independent Canvas 2D backend as the rest of the waterfall and supports the same frequency bounds and live signal labels. The 60-second window is limited by Signal ID's already bounded 60-second live-display cache. If the live session has not yet captured the full history, the missing prehistory remains blank.

## Tests

    node --test tests/test_time_lens.mjs tests/test_live_waterfall.mjs tests/test_signal_overlays.mjs

The pure time-coordinate contract is tested for exact sample/pixel round trips, monotonic nonlinear stretching, linear parity, guard behavior and correct mapping to original spectrogram rows. JavaScript syntax and browser UI checks also cover the new selector, tick labels and module load.

This is a viewport tool, not a signal processing algorithm, not a change in the modulation's timebase and not evidence of a particular protocol.
