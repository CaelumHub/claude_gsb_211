"""
chords.py — Chord recognition and annotation.

The pipeline mirrors a classic template-matching chord recogniser:

  1. Compute a 12-bin chromagram from the STFT (fold spectral energy into
     pitch classes).
  2. Each frame passes a silence/noise gate: an energy floor, a spectral-
     flatness check, and a time-averaged spectral-peak check distinguish
     pitched/harmonic content from silence and white/pink/brown noise.
  3. Surviving frames are compared against a bank of chord templates (major,
     minor, dim, aug, sus2, sus4, maj7, min7, dom7 — all 12 roots) by cosine
     similarity; gated frames are labelled ``N.C.`` ("no chord").
  4. Smooth the label sequence with a majority-vote median filter, then merge
     consecutive frames into labelled segments.

Annotations can then be stored back (the ``/api/analyze/<id>/chords/annotate``
endpoint) so a human can correct the automatic labels.
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Tuple

from . import analysis, dsp

NO_CHORD = "N.C."

# --------------------------------------------------------------------------- #
# "No chord" (silence / noise) rejection thresholds
# --------------------------------------------------------------------------- #
# A frame is considered silent when its windowed spectral RMS is quieter than
# this absolute level *or* more than this many dB below the loudest frame.
_SILENCE_ABS_DB = -90.0
_SILENCE_REL_DB = -35.0

# Noise gate on the *time-averaged* spectrum.  Tonal/voiced content keeps its
# narrow spectral peaks when several frames are averaged; noise averages into a
# smooth envelope.  Threshold chosen between worst-case noise (~0.31 for white/
# pink, ~0.05 for brown) and weakest pitched signal measured (KS pluck ~0.54).
_NOISE_PEAK_RATIO = 0.40
# Per-frame flatness immediately rules out white noise without temporal look-back.
_FLATNESS_MAX = 0.30
# Frames used for the averaged-spectrum gate (~0.5 s at the default hop).
_CONTEXT_FRAMES = 43
_PEAK_FMIN = 40.0
_PEAK_FMAX = 4000.0

# Bump when recognition semantics change; app.py treats a version mismatch in
# the cached result as a cache miss.
CHORD_VERSION = 2

_QUALITY_SUFFIX = {
    "maj": "",
    "min": "m",
    "dim": "dim",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "maj7": "maj7",
    "min7": "m7",
    "dom7": "7",
}

_QUALITY_LONG = {
    "maj": "major",
    "min": "minor",
    "dim": "diminished",
    "aug": "augmented",
    "sus2": "suspended 2nd",
    "sus4": "suspended 4th",
    "maj7": "major 7th",
    "min7": "minor 7th",
    "dom7": "dominant 7th",
}

_TEMPLATES = {
    "maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "min": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "aug": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    "sus2": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    "min7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    "dom7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
}


def _rotate(vec: List[float], n: int) -> List[float]:
    return vec[n:] + vec[:n]


def _norm(vec: List[float]) -> List[float]:
    s = math.sqrt(sum(x * x for x in vec))
    return [x / s for x in vec] if s > 1e-12 else vec


def build_templates() -> List[Dict]:
    """Return the chord template bank (unit-normalised 12-D vectors)."""
    out = []
    for quality, vec in _TEMPLATES.items():
        for root in range(12):
            name = dsp.NOTE_NAMES[root]
            label = name + _QUALITY_SUFFIX[quality]
            out.append({
                "label": label,
                "root": name,
                "quality": quality,
                "quality_long": _QUALITY_LONG[quality],
                "vector": _norm(_rotate(vec, root)),
            })
    return out


_TEMPLATE_BANK = build_templates()


def _best_chord(chroma: List[float]) -> Tuple[str, float]:
    best_label = NO_CHORD
    best_sim = 0.0
    for t in _TEMPLATE_BANK:
        sim = sum(a * b for a, b in zip(chroma, t["vector"]))
        if sim > best_sim:
            best_sim = sim
            best_label = t["label"]
    return best_label, best_sim


def _frame_rms_db(mag: List[float], nfft: int) -> float:
    """Windowed-frame RMS in dB, derived from the STFT magnitude (Parseval)."""
    rms = math.sqrt(2.0 * sum(v * v for v in mag) / nfft ** 2)
    return 20.0 * math.log10(rms) if rms > 1e-12 else -240.0


def _avg_peak_ratio(avg: List[float], k0: int, k1: int) -> float:
    """Fraction of energy sitting at local spectral peaks of a time-averaged
    spectrum (40 Hz–4 kHz).  Near 0 for noise, high for pitched/harmonic sound.
    """
    total = sum(v * v for v in avg[k0:k1])
    if total <= 1e-12:
        return 0.0
    peak_e = 0.0
    for k in range(k0 + 1, k1 - 1):
        if avg[k] > avg[k - 1] and avg[k] >= avg[k + 1]:
            peak_e += avg[k] * avg[k]
    return peak_e / total


def _mode_smooth(labels: List[str], width: int) -> List[str]:
    """Majority-vote smoothing of a categorical label sequence."""
    n = len(labels)
    if width < 1 or n == 0:
        return list(labels)
    half = width // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = labels[lo:hi]
        out.append(max(set(window), key=window.count))
    return out


def recognize(path: str, nfft: int = 2048, hop: int = 512,
              smooth_width: int = 9) -> Dict:
    """Run the full chord-recognition pipeline on a file.

    Frames that are silent (energy gate) or contain no pitched/harmonic content
    (spectral-flatness and time-averaged peak-ratio gates) are labelled
    ``N.C.`` ("no chord") instead of forcing a template match.

    The file is streamed twice so memory stays bounded.  The first pass only
    measures the loudest windowed-RMS level in the time domain (cheap — no
    FFT); the second pass computes the STFT and keeps just a ~0.5 s sliding
    spectral window plus one 12-D chroma vector and label per output frame.
    """
    # First pass — loudest Hann-windowed RMS level (relative-silence reference).
    win = dsp.window("hann", nfft)
    peak_level = -240.0
    sr = 44100
    for frame, sr in analysis.stream_windows(path, nfft, hop):
        energy = sum((frame[k] * win[k]) ** 2 for k in range(nfft))
        if energy > 0.0:
            rms = math.sqrt(energy / nfft)
            if rms > 1e-12:
                peak_level = max(peak_level, 20.0 * math.log10(rms))

    bins = nfft // 2 + 1
    half = _CONTEXT_FRAMES // 2
    k0 = max(1, int(_PEAK_FMIN * nfft / sr))
    k1 = min(bins, int(_PEAK_FMAX * nfft / sr))

    # Second pass.  `window` holds the spectra of the symmetric neighbourhood
    # around the current frame [i-half, min(i+half, end)]; spec_sum is their
    # bin-wise sum feeding the time-averaged noise gate.
    src = iter(analysis.stream_stft(path, nfft, hop))
    window: collections.deque = collections.deque()
    spec_sum = [0.0] * bins
    head = 0                    # global frame index of window[0]

    def _admit() -> bool:
        item = next(src, None)
        if item is None:
            return False
        m, _sr = item
        window.append(m)
        for k in range(bins):
            spec_sum[k] += m[k]
        return True

    # Prime the look-ahead side of the first window.
    while len(window) <= half:
        if not _admit():
            break

    chroma: List[List[float]] = []
    labels: List[str] = []
    confidence: List[float] = []
    i = 0

    while window and i - head < len(window):
        mag = window[i - head]
        ch = _frame_chroma(mag, sr, nfft)
        chroma.append(ch)
        level = _frame_rms_db(mag, nfft)

        # Gate 1 — silence: digital zero, below the absolute floor, or far
        # below the file's own loudest frame.
        if level <= _SILENCE_ABS_DB or level - peak_level <= _SILENCE_REL_DB:
            labels.append(NO_CHORD)
            confidence.append(0.0)
        # Gate 2 — white/flat-spectrum noise on the single frame.
        elif dsp.spectral_flatness(mag) >= _FLATNESS_MAX:
            labels.append(NO_CHORD)
            confidence.append(0.0)
        # Gate 3 — noise vs pitched content on the local time-averaged spectrum:
        # noise fluctuations average into a smooth envelope while sustained
        # harmonic peaks survive averaging.
        elif _avg_peak_ratio(spec_sum, k0, k1) < _NOISE_PEAK_RATIO:
            labels.append(NO_CHORD)
            confidence.append(0.0)
        else:
            label, sim = _best_chord(ch)
            labels.append(label)
            confidence.append(sim)

        # Slide the window to the next centre frame.
        _admit()
        i += 1
        while window and head < i - half:
            old = window.popleft()
            for k in range(bins):
                spec_sum[k] -= old[k]
            head += 1

    nframes = len(labels)
    if nframes == 0:
        return {"version": CHORD_VERSION, "chroma": [], "times": [], "labels": [],
                "confidence": [], "segments": [], "tonal_ratio": 0.0}

    labels = _mode_smooth(labels, smooth_width)

    times = [j * hop / sr for j in range(nframes)]
    segments = _merge_segments(times, labels, confidence, hop / sr)
    tonal_ratio = sum(1 for l in labels if l != NO_CHORD) / nframes

    return {
        "version": CHORD_VERSION,
        "sr": sr,
        "times": times,
        "chroma": chroma,
        "labels": labels,
        "confidence": confidence,
        "segments": segments,
        "tonal_ratio": round(tonal_ratio, 4),
    }


def _frame_chroma(mag: List[float], sr: float, nfft: int) -> List[float]:
    c = [0.0] * 12
    bins = len(mag)
    for k in range(1, bins):
        f = k * sr / nfft
        m = dsp.hz_to_midi(f)
        if m is None:
            continue
        c[int(round(m)) % 12] += mag[k]
    s = sum(c)
    return [v / s for v in c] if s > 1e-12 else c


def _merge_segments(times: List[float], labels: List[str],
                    confidence: List[float], step: float) -> List[Dict]:
    """Merge consecutive equal labels into (start, end, label) segments.

    A segment's confidence is the mean cosine similarity of its frames (0 for
    ``N.C.`` segments, which carry no chord hypothesis).
    """
    segments = []
    if not labels:
        return segments

    def _flush(start: float, end: float, label: str, i0: int, i1: int) -> None:
        if label == NO_CHORD:
            conf = 0.0
        else:
            vals = [confidence[j] for j in range(i0, i1)
                    if confidence[j] > 0.0]
            conf = sum(vals) / len(vals) if vals else 0.0
        segments.append({"start": round(start, 3), "end": round(end, 3),
                         "label": label, "confidence": round(conf, 4)})

    start = times[0]
    prev = labels[0]
    i0 = 0
    for i in range(1, len(labels)):
        if labels[i] != prev:
            _flush(start, times[i - 1] + step, prev, i0, i)
            start = times[i]
            prev = labels[i]
            i0 = i
    _flush(start, times[-1] + step, prev, i0, len(labels))
    return [s for s in segments if s["end"] - s["start"] >= step / 2]


CHORD_LABELS = [t["label"] for t in _TEMPLATE_BANK]
