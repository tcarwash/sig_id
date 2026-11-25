# WebSDR / KiwiSDR / OpenWebRX live input

Signal ID can receive audio from a browser-based remote SDR and send it through
the **same live capture, waterfall, detector, classifier, recording, and decoder
pipeline** used by a local radio audio input.

## Connect from the app

1. Open **Settings → Online receiver**.
2. Choose Northern Utah WebSDR, University of Twente, a receiver directory, or
   paste a direct receiver URL.
3. Optionally enter the tuned frequency in MHz. This is saved as capture
   metadata only; it does not tune the remote receiver.
4. Click **Connect & start live**.
5. A dedicated Chrome/Chromium window opens. Choose the frequency, mode,
   filter, and audio level in that receiver's own page.

Signal ID creates a temporary PipeWire/PulseAudio virtual sink and remapped
capture source. It directs audio streams from only that dedicated browser
process into the virtual source; normal desktop audio and other applications
are left alone. The new input is selected automatically and live capture starts.

The receiver page can be a classic WebSDR, KiwiSDR, OpenWebRX, or a receiver
directory that opens one of those sites in another tab/window. The browser
must be able to play the receiver's audio. Some receivers require an initial
click to start audio or accept their terms.

## Stop and disconnect

- **Stop live** ends the current Signal ID capture but leaves the browser
  connected, so you can tune and resume without rebuilding the audio route.
- **Disconnect** closes the dedicated browser process and unloads the virtual
  sink/source. Stop a live capture using that source first.
- Closing the browser manually changes its status to *Browser closed*; use
  **Disconnect** to remove the remaining virtual audio devices.
- Signal ID also closes the browser and removes the audio route during a clean
  application shutdown.

## Requirements

This is a desktop-host feature. The Signal ID server must run as the logged-in
desktop user, not inside a remote-only container or headless server:

- Chrome or Chromium (google-chrome, google-chrome-stable, chromium, or
  chromium-browser); override the executable with SIG_ID_WEBSDR_BROWSER.
- PipeWire with its PulseAudio compatibility server (or PulseAudio) and pactl.
- PortAudio / sounddevice from the optional audio dependency.
- A logged-in graphical session and its PulseAudio/Wayland environment. Signal
  ID discovers the user's XDG_RUNTIME_DIR, PulseAudio socket and Wayland
  display when the service environment omits them.

For a local/LAN receiver (for example, a private OpenWebRX host), explicitly
enable private URLs in the Signal ID server environment:

~~~sh
SIG_ID_WEBSDR_ALLOW_PRIVATE=1
~~~

Public receiver URLs are accepted by default. Only http:// and https://
URLs are allowed; embedded credentials, file://, browser-internal schemes,
localhost and private IP literals are rejected unless private URLs are enabled.

## Receiver directories

- [WebSDR.org directory](https://www.websdr.org/)
- [KiwiSDR public receiver map](https://rx.linkfanel.net/)

The receiver list is intentionally a starting point, not a promise that a
station is online, available, or supports a particular band. Directory entries
are external sites and retain their own access controls and user interfaces.

## Receive-only and safety boundaries

This integration only opens a browser receiver and captures its audio. It does
not send commands to the remote receiver, change its dial, control a local rig,
or key PTT. Frequency and mode are set in the receiver's own page. The optional
frequency field is only an annotation for the Signal ID capture.

Rigctl band hopping is explicitly disabled when the selected live input is the
WebSDR virtual source; tune the online receiver in its browser window instead.
