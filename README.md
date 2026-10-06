# EyeMorse: hands-free computer control with gaze and blink Morse code

EyeMorse is a multimodal assistive-technology prototype that runs on an
ordinary laptop webcam. You do not need an eye tracker.

| Modality | Channel | Used for |
|---|---|---|
| **Gaze** (iris + head pose) | continuous | pointing: the mouse cursor follows where you look |
| **Blinks / winks** as Morse code | discrete | selecting (click, drag, scroll) and typing text |
| **Switch / key** (optional) | discrete | the same Morse input from a button, as a backup or as a comparison condition |

Gaze is good at pointing but poor at selecting (Jacob's "Midas touch" problem:
everything you look at would get clicked). Blinks are good at deliberate
selection. EyeMorse uses each channel for what it does well.

---

## 1. Quick start

Requires a webcam and Python 3.10–3.12 (the versions MediaPipe supports best). Windows, macOS and Linux (X11) work.

```bash
cd eye-morse-assist
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m eyemorse --dry-run     # first run: see everything work without touching your mouse
python -m eyemorse               # real control
```

The first run downloads the MediaPipe face model (about 4 MB) and starts a **calibration**:

1. Look at each red dot (9 points) while keeping your head still.
2. Look at 5 more dots. These measure accuracy and are not used for fitting.
3. Keep your eyes open for 2 s, then close them for about 3 s. This sets *your* blink thresholds.

The calibration is saved to `calibration.json`. To redo it, run `--calibrate`
or press `c` in the preview window.

**Operating-system permissions**
* **macOS:** System Settings → Privacy & Security → allow your terminal under
  *Camera*, *Accessibility* and (for `--switch-key`) *Input Monitoring*.
* **Linux:** controlling the mouse needs an X11 session. Under Wayland, log in with "GNOME on Xorg".
* **Windows:** no extra steps.

---

## 2. How to use it

### Modes
| Eye closure | Effect |
|---|---|
| 0.12 – 0.4 s | **dot** `.` |
| 0.4 – 1.2 s | **dash** `-` |
| 1.2 – 3 s | **switch mode** (CURSOR ⇄ TYPE) |
| > 3 s | **pause / resume** everything (rest your eyes safely) |
| eyes open 0.9 s | end of letter: the code is executed |

The preview window shows a live bar while your eyes are closed (`DOT`, `DASH`,
`MODE`, `PAUSE`). Open your eyes when it shows what you want.

### CURSOR mode: the cursor follows your gaze
| Code | Action |
|---|---|
| `-` | left click |
| `--` | double click |
| `-.` | right click |
| `-..` | scroll down |
| `-.-` | scroll up |
| `-.-.` | start / stop drag (mouse button held) |
| `---` | freeze / unfreeze the cursor (read without it moving) |

Every command starts with a **dash**, and leading dots are ignored. A normal,
involuntary blink therefore never clicks anything. You can also enable dwell
clicking with `--dwell-click 1.0`, which clicks after you look at one spot for 1 s.

### TYPE mode: the cursor stays still, blinks type text
Standard ITU Morse for `a–z`, `0–9` and punctuation, plus:

| Code | Action |
|---|---|
| `..--` | space |
| `----` | backspace |
| `.-.-` | enter |

This matches Gboard's Morse keyboard, so skills carry over. Press `h` in
the preview window for an on-screen Morse chart.

### Preview-window keys (for a researcher or caregiver)
`h` help/chart · `c` recalibrate · `m` switch mode · `p` pause · `f` freeze cursor · `q`/`Esc` quit

### Input variants
```bash
python -m eyemorse --input-mode both        # close both eyes (default)
python -m eyemorse --input-mode wink-left   # only left-eye winks count
python -m eyemorse --switch-key f9          # hold F9 / a USB switch as extra Morse input
```
In the **wink** modes, natural blinks are rejected automatically because both
eyes close together. The tradeoff is that some people find winking tiring or
cannot wink at all. If left and right appear swapped on your camera, use the
other option.

---

## 3. How it works

```
 webcam ─► MediaPipe Face Landmarker ─┬─► iris + eye corners + nose ─► gaze features ─► ridge poly. regression ─► One Euro filter ─► cursor
           (478 landmarks, blendshapes)│                                (calibrated)                                      ▲
                                       │                                                     gate + rewind on blink ──────┘
                                       └─► eyeBlinkL/R scores ─► closure score ─► hysteresis ─► duration ─► dot/dash/mode/pause
                                                                    (both | wink)    detector      classifier        │
                                                    optional key/switch press durations ─────────────────────────────┤
                                                                                                                    ▼
                                                                     Morse decoder (letter gap) ─► mode-dependent action ─► OS (pyautogui)
```

* **Face tracking:** MediaPipe Face Landmarker (VIDEO mode) returns 478 landmarks,
  including 10 iris points and blendshape scores (`eyeBlinkLeft/Right`). If
  blendshapes are missing, the eye aspect ratio (EAR) is used instead.
* **Gaze features** (`gaze.py`): the iris centre position relative to the
  eye-corner axis, normalised by eye width, plus a head yaw/pitch proxy from the
  nose tip, normalised by inter-ocular distance. The features are scale-invariant.
* **Mapping:** a 2nd-order polynomial with ridge regularisation, fitted on the
  9-point calibration samples after MAD outlier rejection.
* **Smoothing:** One Euro filter, which gives low jitter when the gaze is still and low lag when it moves.
* **Blink artefacts:** gaze is ignored while either eye is partly closed. At the
  start of a blink, the cursor jumps back to its position 150 ms earlier, which
  undoes the downward drift caused by the closing eyelid.
* **Blink detection:** hysteresis thresholds (personalised during calibration)
  turn the closure score into timed closures. A duration classifier labels each
  one. With `--adaptive-timing`, the dot/dash boundary follows the user's own
  rhythm: it is set to the midpoint of their recent mean dot and mean dash durations.

All interaction logic (`app.EyeMorseApp`) is camera-free and unit-tested
(`python -m pytest`).

---

## 4. Using it for your research paper

### Every session is logged
`logs/session_YYYYMMDD-HHMMSS/` contains:
* `config.json`: every parameter used (for reproducibility)
* `events.csv`: each closure (duration, source, symbol), action and mode change, with timestamps
* `calibration.json`: the gaze model and **validation accuracy** (px, and degrees if you pass screen size and distance)
* `summary.json`: session totals (symbols, characters, clicks, unknown codes, mode switches)
* `frames.csv` (with `--log-frames`): per-frame blink scores, closure and cursor, for offline analysis or plots

### Text-entry experiment mode
```bash
python -m eyemorse --trial phrases/practice.txt
```
The preview shows a target phrase. The participant blinks it in and ends it
with `.-.-` (enter). Trial text is **not** typed into other apps unless you pass
`--trial-send`. For each phrase, `trials.json` records the following metrics,
which are standard in text-entry research:

| Metric | Definition |
|---|---|
| **WPM** | (\|T\| − 1) / seconds × 60 / 5, timed from the first dot/dash |
| **Error rate** | MSD(P, T) / max(\|P\|, \|T\|) × 100 (minimum string distance) |
| **KSPC** | Morse symbols / characters (input efficiency, including corrections) |
| backspaces | number of corrections |

### Pointing accuracy
Add `--screen-width-cm 34.5 --viewing-distance-cm 60` (measure your own) to
report calibration error in **degrees of visual angle**. This is the usual unit
in gaze-tracking papers.

### Experiment ideas (independent variables)
* Input channel: `--input-mode both` vs `wink-left` vs `--switch-key` (switch as baseline)
* Fixed vs adaptive timing: `--adaptive-timing`
* Timing parameters: `--dash-threshold`, `--letter-gap` (speed vs. error tradeoff)
* Selection method: blink Morse clicks vs `--dwell-click 0.8`
* Learning effect: repeat trial sessions over several days

Save each condition's parameters with `--save-config condA.json` and load them with `--config condA.json`.

### Suggested background reading
Check each citation yourself before using it in your paper.
* Jacob, R. J. K. (1990). *What you look at is what you get: eye movement-based interaction techniques.* CHI '90. (The Midas touch problem.)
* Soukupová, T. & Čech, J. (2016). *Real-time eye blink detection using facial landmarks.* CVWW. (Eye aspect ratio.)
* Casiez, G., Roussel, N. & Vogel, D. (2012). *1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.* CHI '12.
* Soukoreff, R. W. & MacKenzie, I. S. (2003). *Metrics for text entry research: an evaluation of MSD and KSPC, and a new unified error metric.* CHI '03.
* MacKenzie, I. S. & Soukoreff, R. W. (2003). *Phrase sets for evaluating text entry techniques.* CHI EA '03.
* Ablavatski, A. et al. (2020). *Real-time pupil tracking from monocular video for digital puppetry.* (MediaPipe Iris.)

---

## 5. Tuning and troubleshooting
| Problem | Try |
|---|---|
| Natural blinks type `e` | raise `--min-dot 0.18`, or use `--input-mode wink-left` |
| Dots read as dashes (or the reverse) | adjust `--dash-threshold`, or use `--adaptive-timing` |
| Letters end before you finish them | raise `--letter-gap 1.3` |
| Cursor too jittery | lower `--smoothing-min-cutoff 0.3` |
| Cursor too laggy | raise `--smoothing-beta 0.02` |
| Cursor is offset | recalibrate (`c`). Good, even, front lighting helps a lot. Keep your head still. |
| "NO FACE DETECTED" | improve lighting, centre your face, try `--camera 1` |

### Limitations (worth discussing in the paper)
* Webcam gaze accuracy is coarse, typically a few degrees. Small targets are hard
  to hit, so enlarge them in your OS settings (display scaling, magnifier).
* Large head movements after calibration reduce accuracy. Recalibrate if you move.
* Glasses glare, low light, and droopy eyelids (ptosis) all degrade tracking.
* Blink-based input causes eye fatigue in long sessions. Use the pause gesture.
* This is a research prototype, not a certified medical device.

If you test it with participants, follow your institution's ethics/IRB process.
