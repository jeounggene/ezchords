import os
import uuid
import threading
import tempfile
import shutil
import re
import json
import sqlite3
import time
import urllib.request
import urllib.parse

from flask import Flask, render_template, request, jsonify, send_file
import yt_dlp
import librosa
import numpy as np

app = Flask(__name__)

DB_PATH    = os.path.join(os.path.dirname(__file__), 'cache.db')
AUDIO_DIR  = os.path.join(os.path.dirname(__file__), 'static', 'audio')
os.makedirs(AUDIO_DIR, exist_ok=True)

def _audio_path(video_id: str) -> str:
    return os.path.join(AUDIO_DIR, f'{video_id}.mp3')

# ─────────────────────────────────────────────
# SQLite persistent cache
# ─────────────────────────────────────────────

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute('''
        CREATE TABLE IF NOT EXISTS songs (
            video_id   TEXT PRIMARY KEY,
            title      TEXT,
            key        TEXT,
            bpm        REAL,
            chords     TEXT,   -- JSON
            beat_times TEXT,   -- JSON
            analyzed_at INTEGER
        )
    ''')
    # Migration: add lyrics column for existing databases
    try:
        con.execute('ALTER TABLE songs ADD COLUMN lyrics TEXT')
    except Exception:
        pass  # column already exists
    con.commit()
    con.close()

_init_db()


def _cache_get(video_id: str):
    """Return stored dict or None."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        'SELECT * FROM songs WHERE video_id = ?', (video_id,)
    ).fetchone()
    con.close()
    if row is None:
        return None
    return {
        'status':     'done',
        'video_id':   row['video_id'],
        'title':      row['title'],
        'key':        row['key'],
        'bpm':        row['bpm'],
        'chords':     json.loads(row['chords']),
        'beat_times': json.loads(row['beat_times']),
        'lyrics':     json.loads(row['lyrics'] or '[]'),
    }


def _cache_put(video_id, title, key, bpm, chords, beat_times, lyrics=None):
    con = sqlite3.connect(DB_PATH)
    con.execute('''
        INSERT OR REPLACE INTO songs
            (video_id, title, key, bpm, chords, beat_times, lyrics, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (video_id, title, key, bpm,
          json.dumps(chords), json.dumps(beat_times),
          json.dumps(lyrics or []),
          int(time.time())))
    con.commit()
    con.close()


def _cache_list():
    """Return list of {video_id, title, key, bpm, analyzed_at} dicts."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        'SELECT video_id, title, key, bpm, analyzed_at FROM songs ORDER BY analyzed_at DESC'
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# VTT subtitle parser (for lyrics)
# ─────────────────────────────────────────────

def _split_title(title: str):
    """Split 'Artist - Song Title' → (artist, track). Returns ('', title) if no separator."""
    for sep in [' - ', ' – ', ' — ']:
        if sep in title:
            parts = title.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return '', title.strip()


def _parse_lrc(lrc_text: str) -> list:
    """Parse LRC format → [{text, start, end}, ...]"""
    pattern = re.compile(r'\[(\d+):(\d+(?:\.\d+)?)\](.*)')
    cues = []
    for line in lrc_text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        mins = int(m.group(1))
        secs = float(m.group(2))
        text = m.group(3).strip()
        if text:
            cues.append({'text': text, 'start': round(mins * 60 + secs, 3)})
    # Infer end times from next cue's start
    for i in range(len(cues) - 1):
        cues[i]['end'] = cues[i + 1]['start']
    if cues:
        cues[-1]['end'] = round(cues[-1]['start'] + 8.0, 3)
    return cues


def _fetch_lrclib(title: str, artist: str = '', duration: int = None) -> list:
    """Fetch synced lyrics from lrclib.net. Returns [{text, start, end}, ...] or []."""
    hdrs = {'User-Agent': 'EZChords/1.0'}

    # 1. Exact-match endpoint (needs artist + track)
    if artist:
        params = {'track_name': title, 'artist_name': artist}
        if duration:
            params['duration'] = duration
        req_url = 'https://lrclib.net/api/get?' + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(req_url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    data = json.loads(r.read().decode())
                    synced = data.get('syncedLyrics') or ''
                    if synced:
                        return _parse_lrc(synced)
        except Exception:
            pass

    # 2. Search endpoint
    query = f'{artist} {title}'.strip() if artist else title
    req_url = 'https://lrclib.net/api/search?' + urllib.parse.urlencode({'q': query})
    try:
        req = urllib.request.Request(req_url, headers=hdrs)
        with urllib.request.urlopen(req, timeout=10) as r:
            results = json.loads(r.read().decode()) or []
            for item in results[:5]:
                synced = item.get('syncedLyrics') or ''
                if synced:
                    return _parse_lrc(synced)
    except Exception:
        pass

    return []


def _parse_vtt(path: str):
    """Parse a WebVTT file → [{text, start, end}].
    Handles YouTube auto-caption rolling style (many cues with same start time)."""
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception:
        return []

    def ts_sec(ts):
        parts = ts.strip().replace(',', '.').split(':')
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            return int(parts[0]) * 60 + float(parts[1])
        except (ValueError, IndexError):
            return 0.0

    timing_re = re.compile(r'(\d+:\d+[\d:.]+)\s*-->\s*(\d+:\d+[\d:.]+)')
    tag_re    = re.compile(r'<[^>]+>')

    cues = []
    for block in re.split(r'\n{2,}', content):
        block = block.strip()
        if not block or re.match(r'^(WEBVTT|NOTE|Kind:|Language:)', block):
            continue
        m_time = None
        text_parts = []
        for ln in block.splitlines():
            mt = timing_re.search(ln)
            if mt:
                m_time = mt
                text_parts = []
            elif m_time and ln.strip() and not ln.strip().isdigit():
                text_parts.append(ln.strip())
        if not m_time:
            continue
        raw = ' '.join(text_parts)
        clean = tag_re.sub('', raw).strip()
        clean = re.sub(r'\s+', ' ', clean)
        if clean:
            cues.append({
                'text':  clean,
                'start': round(ts_sec(m_time.group(1)), 3),
                'end':   round(ts_sec(m_time.group(2)), 3),
            })

    if not cues:
        return []

    # YouTube rolling captions: many cues share the same start time, each adding one word.
    # Keep only the last (most complete) cue per start-time group.
    out = []
    i = 0
    while i < len(cues):
        j = i + 1
        while j < len(cues) and abs(cues[j]['start'] - cues[i]['start']) < 0.4:
            j += 1
        best = cues[j - 1]
        if not out or out[-1]['text'] != best['text']:
            out.append(best)
        i = j

    return out


# ─────────────────────────────────────────────
# Chord templates (major, minor, 7th, maj7, min7, sus2, sus4, dim, aug)
# ─────────────────────────────────────────────
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# (suffix, intervals) — used for both simple and extended templates
SIMPLE_INTERVALS = [
    ('',  [0, 4, 7]),        # major
    ('m', [0, 3, 7]),        # minor
]

EXTENDED_INTERVALS = [
    ('7',    [0, 4, 7, 10]),  # dominant 7
    ('m7',   [0, 3, 7, 10]),  # minor 7
    ('maj7', [0, 4, 7, 11]),  # major 7
    ('sus2', [0, 2, 7]),      # sus2
    ('sus4', [0, 5, 7]),      # sus4
    ('dim',  [0, 3, 6]),      # diminished
    ('aug',  [0, 4, 8]),      # augmented
]

ALL_INTERVALS = SIMPLE_INTERVALS + EXTENDED_INTERVALS

# Acoustic weighting: root 1.5×, fifth 1.2×, other intervals 1.0×
ROOT_WEIGHT  = 1.5
FIFTH_WEIGHT = 1.2


def _build_templates(intervals_list):
    """Build weighted chord templates from a list of (suffix, intervals)."""
    templates = {}
    for i, note in enumerate(NOTES):
        for chord_type, intervals in intervals_list:
            t = np.zeros(12)
            for k, iv in enumerate(intervals):
                if k == 0:
                    w = ROOT_WEIGHT   # root
                elif iv in (7, 6, 8):
                    w = FIFTH_WEIGHT  # fifth (or tritone/aug-5th acting as 5th)
                else:
                    w = 1.0
                t[(i + iv) % 12] = w
            t /= np.linalg.norm(t)
            templates[note + chord_type] = t
    return templates


# Simple (major/minor only) — for pass-1
SIMPLE_TEMPLATES  = _build_templates(SIMPLE_INTERVALS)
SIMPLE_CHORDS     = list(SIMPLE_TEMPLATES.keys())
SIMPLE_MATRIX     = np.array([SIMPLE_TEMPLATES[c] for c in SIMPLE_CHORDS])  # (24, 12)

# Full (all 108) — for pass-2
CHORD_TEMPLATES   = _build_templates(ALL_INTERVALS)
ALL_CHORDS        = list(CHORD_TEMPLATES.keys())
TEMPLATE_MATRIX   = np.array([CHORD_TEMPLATES[c] for c in ALL_CHORDS])      # (108, 12)

# Map each extended chord to its simple parent (e.g. "Cm7" → "Cm", "Csus4" → "C")
_EXTENDED_TO_SIMPLE = {}
for i, note in enumerate(NOTES):
    for suffix, _ in EXTENDED_INTERVALS:
        parent = note + ('m' if 'm' in suffix and suffix != 'maj7' else '')
        _EXTENDED_TO_SIMPLE[note + suffix] = parent

# Diatonic chords for each key (major scale degrees I–VII)
_DIATONIC_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_DIATONIC_QUALITIES = ['', 'm', 'm', '', '', 'm', 'dim']


def _diatonic_set(key_idx):
    """Return set of chord names diatonic to the given major key."""
    s = set()
    for offset, quality in zip(_DIATONIC_INTERVALS, _DIATONIC_QUALITIES):
        note = NOTES[(key_idx + offset) % 12]
        s.add(note + quality)
    return s

# ─────────────────────────────────────────────
# In-memory job store  (demo only — not for prod)
# ─────────────────────────────────────────────
_jobs = {}
_jobs_lock = threading.Lock()


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)
        else:
            _jobs[job_id] = dict(kwargs)


def _get_job(job_id):
    with _jobs_lock:
        return dict(_jobs.get(job_id, {'status': 'not_found'}))


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def extract_video_id(url: str):
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


# ─────────────────────────────────────────────
# Chord detection
# ─────────────────────────────────────────────

def _viterbi_decode(template_matrix, chord_list, beat_chroma, self_prob=0.92,
                    emission_bias=None):
    """Run Viterbi HMM over beat_chroma using the given templates.

    template_matrix: (n_chords, 12)
    chord_list:      list of chord names, length n_chords
    beat_chroma:     (12, n_beats)
    emission_bias:   optional (n_chords, 1) additive log-bias per chord
    Returns: list of chord-name per beat.
    """
    n_chords = len(chord_list)
    n_beats  = beat_chroma.shape[1]

    # Cosine similarity → emission scores
    norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    sim = template_matrix @ (beat_chroma / norms)  # (n_chords, n_beats)

    log_emit = np.log(np.clip(sim, 1e-10, None))
    if emission_bias is not None:
        log_emit += emission_bias

    # Transition matrix
    switch_prob = (1.0 - self_prob) / max(n_chords - 1, 1)
    log_self   = np.log(self_prob)
    log_switch = np.log(switch_prob)

    # Forward pass
    viterbi = np.full((n_chords, n_beats), -np.inf)
    backptr = np.zeros((n_chords, n_beats), dtype=int)
    viterbi[:, 0] = np.log(1.0 / n_chords) + log_emit[:, 0]

    for t in range(1, n_beats):
        prev = viterbi[:, t - 1]
        for s in range(n_chords):
            candidates = prev + log_switch
            candidates[s] = prev[s] + log_self
            bp = int(np.argmax(candidates))
            viterbi[s, t] = candidates[bp] + log_emit[s, t]
            backptr[s, t] = bp

    # Back-trace
    path = np.zeros(n_beats, dtype=int)
    path[-1] = int(np.argmax(viterbi[:, -1]))
    for t in range(n_beats - 2, -1, -1):
        path[t] = backptr[path[t + 1], t + 1]

    return [chord_list[ci] for ci in path]


# ─────────────────────────────────────────────
# Essentia-based chord detection (subprocess with Python 3.12)
# ─────────────────────────────────────────────
_VENV312_PYTHON = os.path.join(os.path.dirname(__file__), '.venv312', 'bin', 'python')
_ANALYZE_SCRIPT = os.path.join(os.path.dirname(__file__), 'analyze_chords.py')


def _detect_chords_essentia(audio_path: str):
    """Call the Essentia analysis script in the 3.12 venv. Returns (chords, bpm, key, beat_times)."""
    import subprocess
    result = subprocess.run(
        [_VENV312_PYTHON, _ANALYZE_SCRIPT, audio_path],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Essentia analysis failed: {result.stderr[-500:]}')
    data = json.loads(result.stdout)
    if 'error' in data:
        raise RuntimeError(data['error'])
    return data['chords'], data['bpm'], data['key'], data['beat_times']


def detect_chords(audio_path: str, hop_size: float = 0.5):
    """Return (chords_list, bpm, key_string, beat_times_list).

    Tries Essentia (via Python 3.12 subprocess) first for best accuracy.
    Falls back to librosa CENS + HMM pipeline if Essentia is unavailable.
    """
    # ── Try Essentia first ──
    if os.path.exists(_VENV312_PYTHON) and os.path.exists(_ANALYZE_SCRIPT):
        try:
            return _detect_chords_essentia(audio_path)
        except Exception as e:
            print(f'[EZChords] Essentia failed, falling back to librosa: {e}')

    # ── Fallback: librosa CENS + Viterbi HMM ──
    return _detect_chords_librosa(audio_path, hop_size)


def _detect_chords_librosa(audio_path: str, hop_size: float = 0.5):
    """Fallback librosa-based chord detection.

    Pipeline:
      1) HPSS → harmonic chroma (CENS) + percussive beat tracking
      2) Beat-synchronous chroma
      3) Key detection → diatonic emission bias
      4) Pass-1: Viterbi HMM with 24 major/minor templates
      5) Pass-2: Promote to extended chord only when score is significantly higher
    """
    y, sr = librosa.load(audio_path, mono=True, sr=22050, duration=360)

    hop_length = 2048  # ≈93 ms

    # ── HPSS ──
    y_harm, y_perc = librosa.effects.hpss(y)

    # ── CENS chroma (Chroma Energy Normalized Statistics) ──
    chroma = librosa.feature.chroma_cens(
        y=y_harm, sr=sr, hop_length=hop_length,
        n_chroma=12,
    )

    # ── Beat tracking from percussive ──
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr, hop_length=hop_length)
    tempo = np.asarray(tempo).item()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    # ── Beat-synchronous chroma ──
    beat_chroma = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    n_beats = len(beat_times)
    beat_chroma = beat_chroma[:, :n_beats]

    # ── Key detection ──
    key_idx = int(np.argmax(np.mean(beat_chroma, axis=1)))
    key = NOTES[key_idx]
    diatonic = _diatonic_set(key_idx)

    # ── Key-aware emission bias for simple chords (pass-1) ──
    KEY_BOOST = 0.6  # log-space bonus for diatonic chords
    simple_bias = np.zeros((len(SIMPLE_CHORDS), 1))
    for ci, name in enumerate(SIMPLE_CHORDS):
        if name in diatonic:
            simple_bias[ci, 0] = KEY_BOOST

    # ── Pass-1: Viterbi with 24 major/minor templates ──
    simple_path = _viterbi_decode(SIMPLE_MATRIX, SIMPLE_CHORDS, beat_chroma,
                                  self_prob=0.92, emission_bias=simple_bias)

    # ── Pass-2: Try to promote each beat to an extended chord ──
    # Only upgrade if the extended template scores ≥ PROMOTE_THRESH higher
    PROMOTE_THRESH = 0.12
    norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    bc_normed = beat_chroma / norms
    full_sim = TEMPLATE_MATRIX @ bc_normed  # (108, n_beats)

    final_path = []
    for bi, simple_name in enumerate(simple_path):
        simple_score = full_sim[ALL_CHORDS.index(simple_name), bi]
        # Check all extended variants that share the same root
        best_ext_name  = simple_name
        best_ext_score = simple_score
        for ext_name, parent in _EXTENDED_TO_SIMPLE.items():
            if parent == simple_name:
                ext_score = full_sim[ALL_CHORDS.index(ext_name), bi]
                if ext_score > best_ext_score + PROMOTE_THRESH:
                    best_ext_name  = ext_name
                    best_ext_score = ext_score
        final_path.append(best_ext_name)

    # ── Merge consecutive identical chords ──
    merged = []
    for i, chord_name in enumerate(final_path):
        start = beat_times[i]
        end = beat_times[i + 1] if i + 1 < len(beat_times) else start + 0.5
        if merged and merged[-1]['chord'] == chord_name:
            merged[-1]['end'] = round(end, 3)
        else:
            merged.append({'chord': chord_name, 'start': round(start, 3), 'end': round(end, 3)})

    return merged, tempo, key, beat_times


# ─────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────

def _process_job(job_id: str, url: str):
    tmp_dir = tempfile.mkdtemp()
    try:
        _set_job(job_id, status='processing', progress=5, message='Initializing…')

        video_id = extract_video_id(url)
        mp3_dest  = _audio_path(video_id)

        # ── Download (skip if we already have the audio on disk) ──
        if os.path.exists(mp3_dest):
            _set_job(job_id, message='Audio already cached, skipping download…', progress=30)
            title_from_cache = (_cache_get(video_id) or {}).get('title', 'Unknown Song')
            title = title_from_cache
        else:
            ydl_opts = {
                'format': (
                    'bestaudio[abr<=96]/bestaudio[abr<=128]'
                    '/bestaudio[abr<=160]/bestaudio/best'
                ),
                # Download to tmp; we'll copy the mp3 to AUDIO_DIR after
                'outtmpl': os.path.join(tmp_dir, 'audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '128',
                }],
                'socket_timeout': 60,
                'retries': 8,
                'fragment_retries': 8,
                'file_access_retries': 3,
                'extractor_retries': 3,
                'concurrent_fragment_downloads': 4,
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
            }

            _set_job(job_id, message='Downloading audio from YouTube…', progress=10)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info  = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown Song')

            # Locate the mp3 yt-dlp produced
            tmp_mp3 = os.path.join(tmp_dir, 'audio.mp3')
            if not os.path.exists(tmp_mp3):
                candidates = [f for f in os.listdir(tmp_dir)
                              if os.path.isfile(os.path.join(tmp_dir, f))]
                if not candidates:
                    raise RuntimeError('No audio file found after download.')
                tmp_mp3 = os.path.join(tmp_dir, candidates[0])

            # Save permanently so the browser can stream it
            shutil.copy2(tmp_mp3, mp3_dest)

        # ── Fetch lyrics via LRCLIB (then fall back to YT subtitles) ──
        _pre   = _cache_get(video_id)
        lyrics = (_pre.get('lyrics') or []) if _pre else []
        if not lyrics:
            _set_job(job_id, message='Fetching lyrics…', progress=38)
            artist, track = _split_title(title)
            # Try to use yt-dlp metadata if available
            if 'info' in dir():
                artist = info.get('artist') or info.get('uploader') or artist
                track  = info.get('track')  or track or title
                dur    = info.get('duration')
            else:
                dur = None
            lyrics = _fetch_lrclib(track or title, artist, dur)
            if not lyrics:
                # Fallback: try YouTube auto-captions
                subs_dir = os.path.join(tmp_dir, 'subs')
                os.makedirs(subs_dir, exist_ok=True)
                try:
                    ydl_sub_opts = {
                        'writesubtitles':    True,
                        'writeautomaticsub': True,
                        'subtitlesformat':   'vtt',
                        'subtitleslangs':    ['en', 'en-US', 'en-GB'],
                        'skip_download':     True,
                        'outtmpl':           os.path.join(subs_dir, '%(id)s.%(ext)s'),
                        'quiet':             True,
                        'no_warnings':       True,
                        'noplaylist':        True,
                    }
                    with yt_dlp.YoutubeDL(ydl_sub_opts) as ydl:
                        ydl.download([url])
                    vtt_files = [os.path.join(subs_dir, f)
                                 for f in os.listdir(subs_dir) if f.endswith('.vtt')]
                    if vtt_files:
                        lyrics = _parse_vtt(vtt_files[0])
                except Exception as e:
                    print(f'[lyrics] subtitle fallback failed: {e}')

        _set_job(job_id, message='Analyzing chords (this takes ~30–60 s)…', progress=40)

        # If chord data already in SQLite (e.g. audio-only re-download), skip analysis
        existing = _cache_get(video_id)
        if existing:
            # Persist newly-fetched lyrics into the cache if they were missing
            if lyrics and not (existing.get('lyrics') or []):
                _cache_put(video_id, existing['title'], existing['key'], existing['bpm'],
                           existing['chords'], existing['beat_times'], lyrics)
            _set_job(
                job_id,
                status='done',
                progress=100,
                title=existing['title'],
                chords=existing['chords'],
                bpm=existing['bpm'],
                key=existing['key'],
                beat_times=existing['beat_times'],
                lyrics=lyrics or existing.get('lyrics') or [],
                video_id=video_id,
                has_audio=True,
            )
        else:
            chords, bpm, key, beat_times = detect_chords(mp3_dest)

            # ── Persist metadata to SQLite ────────────────────────────────────
            _cache_put(video_id, title, key, round(bpm, 1), chords, beat_times, lyrics)

            _set_job(
                job_id,
                status='done',
                progress=100,
                title=title,
                chords=chords,
                bpm=round(bpm, 1),
                key=key,
                beat_times=beat_times,
                lyrics=lyrics,
                video_id=video_id,
                has_audio=True,
            )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set_job(job_id, status='error', message=f'Analysis failed: {exc}')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _fetch_lyrics_job(video_id: str, url: str):
    """Background job: fetch lyrics via LRCLIB (fallback: YT captions) and update SQLite."""
    try:
        cached = _cache_get(video_id)
        if not cached:
            return
        title  = cached.get('title', '')
        artist, track = _split_title(title)
        lyrics = _fetch_lrclib(track or title, artist)
        if not lyrics:
            # Fallback: YouTube auto-captions
            subs_dir = tempfile.mkdtemp()
            try:
                ydl_sub_opts = {
                    'writesubtitles': True, 'writeautomaticsub': True,
                    'subtitlesformat': 'vtt', 'subtitleslangs': ['en', 'en-US', 'en-GB'],
                    'skip_download': True,
                    'outtmpl': os.path.join(subs_dir, '%(id)s.%(ext)s'),
                    'quiet': True, 'no_warnings': True, 'noplaylist': True,
                }
                with yt_dlp.YoutubeDL(ydl_sub_opts) as ydl:
                    ydl.download([url])
                vtt_files = [os.path.join(subs_dir, f)
                             for f in os.listdir(subs_dir) if f.endswith('.vtt')]
                if vtt_files:
                    lyrics = _parse_vtt(vtt_files[0])
            finally:
                shutil.rmtree(subs_dir, ignore_errors=True)
        if lyrics:
            _cache_put(video_id, cached['title'], cached['key'], cached['bpm'],
                       cached['chords'], cached['beat_times'], lyrics)
    except Exception as e:
        print(f'[lyrics] background fetch failed: {e}')


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL — could not extract video ID.'}), 400

    # ── 1. Check persistent SQLite cache ────────────────────────────────────
    cached = _cache_get(video_id)
    if cached:
        if os.path.exists(_audio_path(video_id)):
            # Chord data + audio both ready — return instantly
            cached['has_audio'] = True
            job_id = 'cache:' + video_id
            _set_job(job_id, **cached)
            return jsonify({'job_id': job_id, 'cached': True})
        else:
            # Chord data exists but mp3 missing (analyzed before audio-saving was added).
            # Start a lightweight background job that only downloads the audio.
            job_id = str(uuid.uuid4())
            _set_job(job_id, status='processing', progress=0,
                     message='Downloading audio (chord data already cached)…')
            t = threading.Thread(target=_process_job, args=(job_id, url), daemon=True)
            t.start()
            return jsonify({'job_id': job_id, 'video_id': video_id, 'cached': False})

    # ── 2. Check in-memory jobs (already running or finished this session) ─
    with _jobs_lock:
        for jid, job in _jobs.items():
            if job.get('status') == 'done' and job.get('video_id') == video_id:
                return jsonify({'job_id': jid, 'cached': True})

    # ── 3. Start a new background job ─────────────────────────────────────
    job_id = str(uuid.uuid4())
    _set_job(job_id, status='processing', progress=0, message='Queued…')

    t = threading.Thread(target=_process_job, args=(job_id, url), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'video_id': video_id, 'cached': False})


@app.route('/api/cached')
def cached_songs():
    """List all previously analyzed songs from the persistent cache."""
    return jsonify(_cache_list())


@app.route('/api/audio/<video_id>')
def serve_audio(video_id):
    """Stream the stored mp3 for in-browser playback."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]{11}', video_id):
        return jsonify({'error': 'Invalid video ID.'}), 400
    path = _audio_path(video_id)
    if not os.path.exists(path):
        return jsonify({'error': 'Audio not found.'}), 404
    return send_file(path, mimetype='audio/mpeg', conditional=True)


@app.route('/api/lyrics/<video_id>')
def get_lyrics(video_id):
    """Return stored lyrics. If none exist, kick off a background fetch."""
    if not re.fullmatch(r'[a-zA-Z0-9_-]{11}', video_id):
        return jsonify({'error': 'Invalid video ID.'}), 400
    cached = _cache_get(video_id)
    if not cached:
        return jsonify({'lyrics': [], 'status': 'no_song'}), 404
    lyrics = cached.get('lyrics') or []
    if not lyrics:
        # Trigger a background fetch so next poll will have results
        url = f'https://www.youtube.com/watch?v={video_id}'
        t = threading.Thread(target=_fetch_lyrics_job, args=(video_id, url), daemon=True)
        t.start()
        return jsonify({'lyrics': [], 'status': 'fetching'})
    return jsonify({'lyrics': lyrics, 'status': 'ok'})


@app.route('/api/cached/<video_id>', methods=['DELETE'])
def delete_cached(video_id):
    """Remove a song from the cache."""
    # Sanitize: video IDs are exactly 11 alphanumeric/dash/underscore chars
    if not re.fullmatch(r'[a-zA-Z0-9_-]{11}', video_id):
        return jsonify({'error': 'Invalid video ID.'}), 400
    con = sqlite3.connect(DB_PATH)
    con.execute('DELETE FROM songs WHERE video_id = ?', (video_id,))
    con.commit()
    con.close()
    return jsonify({'deleted': video_id})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    return jsonify(_get_job(job_id))


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
