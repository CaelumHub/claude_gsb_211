"""
chords.py — Chord recognition and annotation.

The pipeline mirrors a classic template-matching chord recogniser:

  1. Compute a 12-bin chromagram from the STFT (fold spectral energy into
     pitch classes over the musical range C2..C7).
  2. Compare each chroma frame against a bank of chord templates (major, minor,
     dim, aug, sus2, sus4, maj7, min7, dom7 — all 12 roots) by cosine
     similarity.
  3. Run a "no-chord" gate per frame so that silence and non-tonal content
     (white/pink/brown noise) is labelled N.C. instead of being force-fit to
     the nearest template.
  4. Smooth the label sequence with a majority-vote median filter, then merge
     consecutive frames into labelled segments (short spurious segments are
     dropped).

Annotations can then be stored back (the ``/api/analyze/<id>/chords/annotate``
endpoint) so a human can correct the automatic labels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from . import analysis, dsp

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

NO_CHORD = "N.C."

# --------------------------------------------------------------------------- #
# No-chord gating thresholds (calibrated against synthetic silence / white /
# pink / brown / narrow-band noise and real tonal material).
# --------------------------------------------------------------------------- #
_CHROMA_MIDI_LO = 36.0   # C2 — below this the STFT has too few bins/octave
_CHROMA_MIDI_HI = 96.0   # C7 — effectively no chord energy above
_TONAL_FLO = 65.0        # C2
_TONAL_FHI = 2093.0      # C7
# Peak/envelope test on a single frame with a narrow neighbourhood
# (catches dense harmony: many closely spaced partials).
_NARROW_HALF = 3
_NARROW_RATIO = 2.5
# Same test on the magnitude averaged over a short window with a wide
# neighbourhood (catches sparse, stationary tones: a pure-sine triad has
# only ~3 peaks, which add coherently over time while noise averages flat).
_WIDE_HALF = 8
_WIDE_RATIO = 3.0
_AVG_WINDOW = 9          # frames (~104 ms at 44.1 kHz/512 hop)
_TONAL_MIN = 0.15        # min fraction of in-band power at harmonic peaks
_SIM_MIN = 0.25          # min cosine similarity vs best chord template
_ABS_RMS_FLOOR = 3.2e-4  # ~ -70 dBFS: quieter frames carry no usable content
_REL_RMS_RATIO = 0.10    # frame RMS must clear 10 % of the track's loud level
_MIN_SEGMENT_FRAMES = 8  # ~0.09 s at 44.1 kHz/512 hop; drops isolated blips

# Mean-square of a Hann window (asymptotic 3/8); Parseval RMS calibration.
_WINDOW_MEAN_SQ = 0.375


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
    best_sim = -1.0
    for t in _TEMPLATE_BANK:
        sim = sum(a * b for a, b in zip(chroma, t["vector"]))
        if sim > best_sim:
            best_sim = sim
            best_label = t["label"]
    return best_label, best_sim


def _smooth_labels(raw_labels: List[str], voiced: List[bool],
                   width: int) -> List[str]:
    """Temporal smoothing that respects the no-chord decision.

    A frame keeps a chord label only when chord frames are the majority in
    its smoothing window; among those chord frames the most common chord
    wins.  Otherwise the frame is N.C. — isolated one- or two-frame matches
    (the residue of random peaks in noise) cannot survive the window.
    """
    n = len(raw_labels)
    if width < 1 or n == 0:
        return list(raw_labels)
    half = width // 2
    out: List[str] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        chord_window = [raw_labels[j] for j in range(lo, hi) if voiced[j]]
        if len(chord_window) * 2 > hi - lo:
            out.append(max(set(chord_window), key=chord_window.count))
        else:
            out.append(NO_CHORD)
    return out


def _frame_rms(mag: List[float], nfft: int) -> float:
    """Time-domain RMS of the windowed frame via Parseval's theorem."""
    power = mag[0] ** 2 + mag[-1] ** 2 + 2 * sum(v * v for v in mag[1:-1])
    return math.sqrt(max(0.0, power)) / (nfft * math.sqrt(_WINDOW_MEAN_SQ))


def _peak_fraction(mag: List[float], sr: float, nfft: int,
                   half: int, ratio: float) -> float:
    """Fraction of in-band power sitting in narrow spectral peaks.

    A bin counts as a peak when its magnitude exceeds ``ratio`` times the
    mean magnitude of its neighbours (excluding itself).  Tonal signals
    concentrate their energy in sharp spectral lines; noise — white, pink,
    brown or spectrally tilted/filtered — has a locally smooth spectrum so
    its fraction stays near zero.
    """
    lo = max(1, int(math.ceil(_TONAL_FLO * nfft / sr)))
    hi = min(nfft // 2, int(math.floor(_TONAL_FHI * nfft / sr)))
    if hi <= lo:
        return 0.0
    csum = [0.0]
    for m in mag:
        csum.append(csum[-1] + m)
    total = 0.0
    peaked = 0.0
    for k in range(lo, hi + 1):
        power = mag[k] * mag[k]
        total += power
        a = max(lo, k - half)
        b = min(hi, k + half)
        neigh_n = max(1, (b - a + 1) - 1)
        env = ((csum[b + 1] - csum[a]) - mag[k]) / neigh_n
        if env > 1e-12 and mag[k] > ratio * env:
            peaked += power
    return peaked / total if total > 1e-12 else 0.0


def _tonal_pass(path: str, sr: float, nfft: int, hop: int) -> List[float]:
    """Streaming tonality evidence with bounded memory.

    Magnitude frames are re-read from disk and a rolling window of
    :data:`_AVG_WINDOW` frames is averaged, so the buffer holds at most that
    many frames regardless of file length.  Per frame the detector combines:

    * a narrow single-frame peak test (dense, closely spaced partials);
    * a wide peak test on the window average (sparse stationary tones —
      their harmonics add coherently while random noise averages flat).
    """
    bins = nfft // 2 + 1
    ring: List[List[float]] = []
    acc = [0.0] * bins
    out: List[float] = []
    for mag, _ in analysis.stream_stft(path, nfft, hop):
        ring.append(mag)
        for k in range(bins):
            acc[k] += mag[k]
        if len(ring) > _AVG_WINDOW:
            old = ring.pop(0)
            for k in range(bins):
                acc[k] -= old[k]
        narrow = _peak_fraction(mag, sr, nfft, _NARROW_HALF, _NARROW_RATIO)
        avg = [v / len(ring) for v in acc]
        wide = _peak_fraction(avg, sr, nfft, _WIDE_HALF, _WIDE_RATIO)
        out.append(max(narrow, wide))
    return out


def recognize(path: str, nfft: int = 2048, hop: int = 512,
              smooth_width: int = 9) -> Dict:
    """Run the full chord-recognition pipeline on a file.

    Frames that contain no audible content, or whose spectrum is non-tonal
    (noise), are labelled :data:`NO_CHORD` with zero confidence rather than
    being force-fit to a chord template.
    """
    chroma: List[List[float]] = []
    energy: List[float] = []
    sr = 44100
    for mag, sr in analysis.stream_stft(path, nfft, hop):
        c, _ = _frame_chroma(mag, sr, nfft)
        chroma.append(c)
        energy.append(_frame_rms(mag, nfft))

    if not chroma:
        return {"chroma": [], "times": [], "labels": [], "confidence": [],
                "segments": [], "has_chord": False}

    # Second pass: tonality evidence with a rolling magnitude average, so
    # only a small window of frames is held in memory regardless of length.
    tonal = _tonal_pass(path, sr, nfft, hop)

    # Adaptive level gate: 10 % of the track's 90th-percentile frame level,
    # but never below the absolute floor.  Scale-invariant for loud/quiet
    # noise; the absolute floor handles near-silence in otherwise quiet files.
    level = sorted(energy)
    p90 = level[int(0.9 * (len(level) - 1))]
    rms_gate = max(_ABS_RMS_FLOOR, _REL_RMS_RATIO * p90)

    raw_labels: List[str] = []
    confidence: List[float] = []
    voiced: List[bool] = []
    for c, rms_v, ton in zip(chroma, energy, tonal):
        label, sim = _best_chord(c)
        is_chord = rms_v >= rms_gate and sim >= _SIM_MIN and ton >= _TONAL_MIN
        voiced.append(is_chord)
        if is_chord:
            raw_labels.append(label)
            confidence.append(sim)
        else:
            raw_labels.append(NO_CHORD)
            confidence.append(0.0)

    labels = _smooth_labels(raw_labels, voiced, smooth_width)

    times = [i * hop / sr for i in range(len(chroma))]
    segments = _merge_segments(times, labels, confidence, hop / sr)
    chord_segments = [s for s in segments if s["label"] != NO_CHORD]
    result = {
        "sr": sr,
        "times": times,
        "chroma": chroma,
        "labels": labels,
        "confidence": confidence,
        "segments": segments,
        "has_chord": bool(chord_segments),
    }
    if chord_segments:
        result["chord_time_ratio"] = round(min(1.0,
            sum(s["end"] - s["start"] for s in chord_segments)
            / (times[-1] + hop / sr)), 4)
    return result


def _frame_chroma(mag: List[float], sr: float,
                  nfft: int) -> Tuple[List[float], float]:
    """Return (normalised 12-bin chroma, raw energy) for one STFT frame.

    Energy is folded only over the musically meaningful range C2..C7: below
    C2 there are too few FFT bins per octave, so low-frequency rumble and
    brown-noise tilt would otherwise masquerade as a chord root.
    """
    c = [0.0] * 12
    for k in range(1, len(mag)):
        f = k * sr / nfft
        m = dsp.hz_to_midi(f)
        if m is None or not (_CHROMA_MIDI_LO <= m <= _CHROMA_MIDI_HI):
            continue
        c[int(round(m)) % 12] += mag[k]
    s = sum(c)
    return ([v / s for v in c] if s > 1e-12 else c), s


def _merge_segments(times: List[float], labels: List[str],
                    confidence: List[float], step: float) -> List[Dict]:
    """Merge consecutive equal labels into (start, end, label) segments.

    A lone segment is always kept (so a very short clip still reports its
    result); when there are several segments, isolated ones shorter than the
    minimum duration are dropped — they are the residue of random peaks in
    noisy frames surviving the vote.
    """
    segments = []
    if not labels:
        return segments
    start = times[0]
    start_i = 0
    prev = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev:
            segments.append({"start": round(start, 3),
                             "end": round(times[i - 1] + step, 3),
                             "label": prev,
                             "confidence": _segment_confidence(
                                 confidence, start_i, i - 1, prev)})
            start = times[i]
            start_i = i
            prev = labels[i]
    segments.append({"start": round(start, 3),
                     "end": round(times[-1] + step, 3),
                     "label": prev,
                     "confidence": _segment_confidence(
                         confidence, start_i, len(labels) - 1, prev)})
    if len(segments) <= 1:
        return segments
    min_dur = step * _MIN_SEGMENT_FRAMES
    kept = [s for s in segments if (s["end"] - s["start"]) >= min_dur]
    # If the filter removed everything, trust the longest survivor rather
    # than reporting nothing at all.
    return kept or [max(segments, key=lambda s: s["end"] - s["start"])]


def _segment_confidence(confidence: List[float], lo: int, hi: int,
                        label: str) -> float:
    """Median per-frame similarity over the segment (0.0 for no-chord)."""
    if label == NO_CHORD:
        return 0.0
    vals = sorted(confidence[lo:hi + 1])
    return round(vals[len(vals) // 2], 3) if vals else 0.0


CHORD_LABELS = [t["label"] for t in _TEMPLATE_BANK]
