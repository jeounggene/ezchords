#!/usr/bin/env python3
"""Chord & beat analysis using Essentia.

Called as a subprocess by app.py (needs Python 3.12 + essentia-tensorflow).
Reads an audio file path from argv[1], writes JSON to stdout:
  { "chords": [...], "bpm": float, "key": str, "beat_times": [...] }
"""
import sys
import json
import numpy as np
from essentia.standard import (
    MonoLoader, RhythmExtractor2013,
    HPCP, ChordsDetection, Key,
    FrameGenerator, Windowing, Spectrum, SpectralPeaks,
)

FRAME_SIZE = 4096
HOP_SIZE   = 2048
SR         = 44100


def analyze(audio_path):
    # Load audio as mono @ 44100
    audio = MonoLoader(filename=audio_path, sampleRate=SR)()

    # ── Beat tracking ──
    rhythm = RhythmExtractor2013(method='multifeature')
    bpm, beats, beats_confidence, _, beats_intervals = rhythm(audio)
    beat_times = beats.tolist()

    # ── HPCP (Harmonic Pitch Class Profile) per frame ──
    windowing = Windowing(type='blackmanharris62')
    spectrum  = Spectrum()
    peaks     = SpectralPeaks(
        orderBy='magnitude',
        magnitudeThreshold=0.00001,
        maxPeaks=60,
        minFrequency=20,
        maxFrequency=3500,
        sampleRate=SR,
    )
    hpcp = HPCP(
        size=36,           # 3 bins per semitone for finer resolution
        referenceFrequency=440,
        harmonics=8,
        bandPreset=True,
        minFrequency=20,
        maxFrequency=3500,
        weightType='cosine',
        nonLinear=True,
        windowSize=1.0,
        sampleRate=SR,
    )

    hpcps = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                startFromZero=True):
        spec = spectrum(windowing(frame))
        freqs, mags = peaks(spec)
        h = hpcp(freqs, mags)
        hpcps.append(h)

    hpcps = np.array(hpcps)

    # ── Chord detection (Essentia's built-in template + HMM) ──
    chords_detect = ChordsDetection(hopSize=HOP_SIZE, sampleRate=SR, windowSize=2)
    chord_labels, chord_strengths = chords_detect(hpcps)

    # ── Key detection ──
    # Use 12-bin HPCP for key estimation
    hpcp12 = HPCP(size=12, referenceFrequency=440, harmonics=8,
                   bandPreset=True, sampleRate=SR)
    hpcps12 = []
    for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE,
                                startFromZero=True):
        spec = spectrum(windowing(frame))
        freqs, mags = peaks(spec)
        h = hpcp12(freqs, mags)
        hpcps12.append(h)
    hpcps12 = np.array(hpcps12)
    avg_hpcp = np.mean(hpcps12, axis=0)

    key_algo = Key(profileType='bgate')
    key_name, scale, key_strength, _ = key_algo(avg_hpcp)
    key_str = key_name if scale == 'major' else key_name + 'm'

    # ── Convert frame-level chords to timed segments ──
    frame_duration = HOP_SIZE / SR
    merged = []
    for i, label in enumerate(chord_labels):
        start = round(i * frame_duration, 3)
        end   = round((i + 1) * frame_duration, 3)
        # Normalise Essentia's naming: "A#" → "Bb",  keep "m" suffix
        chord = _normalise_chord(label)
        if merged and merged[-1]['chord'] == chord:
            merged[-1]['end'] = end
        else:
            merged.append({'chord': chord, 'start': start, 'end': end})

    return {
        'chords':     merged,
        'bpm':        round(float(bpm), 1),
        'key':        key_str,
        'beat_times': [round(b, 3) for b in beat_times],
    }


# Essentia uses sharps; map to preferred enharmonic
_ENHARMONIC = {
    'A#': 'Bb', 'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab',
}


def _normalise_chord(label):
    """Normalise Essentia chord label to our preferred naming."""
    if label in ('', 'N', 'None'):
        return 'N'
    # E.g. "A#m" → root="A#", suffix="m"
    if len(label) >= 2 and label[1] == '#':
        root, suffix = label[:2], label[2:]
    else:
        root, suffix = label[0], label[1:]
    root = _ENHARMONIC.get(root, root)
    return root + suffix


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: analyze_chords.py <audio_path>'}))
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result))
