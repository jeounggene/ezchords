import os
import uuid
import threading
import tempfile
import shutil
import re
import json
import sqlite3
import time
import subprocess
import urllib.request
import urllib.parse

# Fix macOS Python SSL (missing root certs)
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except ImportError:
    pass

from flask import Flask, render_template, request, jsonify, send_file
from pytubefix import YouTube
import librosa
import numpy as np

_YT_CLIENTS = ['ANDROID_VR', 'IOS', 'ANDROID', 'WEB', 'TV', 'WEB_MUSIC', 'MWEB']
_YT_MAX_RETRIES = 2          # retry the entire client chain this many times
_YT_BASE_DELAY  = 5          # seconds before first retry (doubled each round)

app = Flask(__name__)

DB_PATH    = os.path.join(os.path.dirname(__file__), 'cache.db')
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _upload_audio_path(upload_id: str) -> str:
    return os.path.join(UPLOAD_DIR, f'{upload_id}.mp3')


def _clean_title(name: str) -> str:
    title = re.sub(r'\.[A-Za-z0-9]{1,6}$', '', (name or '').strip())
    title = re.sub(r'[_\-]+', ' ', title).strip()
    return title[:120] if title else 'Uploaded Audio'


def _youtube_metadata(url: str):
    last_err = None
    for client in _YT_CLIENTS:
        try:
            yt = YouTube(url, client=client)
            _ = yt.title
            return {
                'title': yt.title or '',
                'duration': int(yt.length or 0),
                'video_id': getattr(yt, 'video_id', '') or extract_video_id(url),
            }
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f'Could not fetch source metadata: {last_err}')

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


def _process_upload_job(job_id: str, source_path: str, source_name: str):
    """Analyze a user-uploaded audio file, persist only chord data, then delete audio."""
    temp_mp3 = _upload_audio_path(job_id)
    try:
        _set_job(job_id, status='processing', progress=5, message='Preparing uploaded audio…')

        if not shutil.which('ffmpeg'):
            raise RuntimeError('FFmpeg is required to process uploaded files.')

        subprocess.run(
            [
                'ffmpeg', '-i', source_path,
                '-vn', '-ar', '44100', '-ac', '2', '-b:a', '128k',
                temp_mp3, '-y',
            ],
            capture_output=True, timeout=180,
        )
        if not os.path.exists(temp_mp3):
            raise RuntimeError('Failed to convert uploaded file to mp3.')

        _set_job(job_id, status='processing', progress=40, message='Analyzing chords…')
        chords, bpm, key, beat_times = detect_chords(temp_mp3)

        source_url = (_get_job(job_id).get('source_url') or '').strip()
        video_id = extract_video_id(source_url) if source_url else None
        title = _clean_title(source_name)

        if video_id:
            _cache_put(video_id, title, key, round(bpm, 1), chords, beat_times, lyrics=[])

        _set_job(
            job_id,
            status='done',
            progress=100,
            title=title,
            chords=chords,
            bpm=round(bpm, 1),
            key=key,
            beat_times=beat_times,
            lyrics=[],
            video_id=video_id,
            source_url=source_url,
            has_audio=False,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _set_job(job_id, status='error', message=f'Upload analysis failed: {exc}')
    finally:
        try:
            if os.path.exists(source_path):
                os.remove(source_path)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception:
            pass


def _parse_srt(srt_text: str) -> list:
    """Parse SRT text → [{text, start, end}, ...]."""
    blocks = re.split(r'\n{2,}', srt_text.strip())
    cues = []
    ts_re = re.compile(r'(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)')
    for block in blocks:
        lines = block.strip().splitlines()
        m = None
        text_parts = []
        for ln in lines:
            mt = ts_re.search(ln)
            if mt:
                m = mt
                text_parts = []
            elif m and ln.strip() and not ln.strip().isdigit():
                text_parts.append(re.sub(r'<[^>]+>', '', ln.strip()))
        if not m or not text_parts:
            continue
        def _ts(s):
            p = s.replace(',', '.').split(':')
            return round(int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2]), 3)
        text = re.sub(r'[♪♫]', '', ' '.join(text_parts)).strip()
        if text:
            cues.append({'text': text, 'start': _ts(m.group(1)), 'end': _ts(m.group(2))})
    return cues


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/source-metadata')
def source_metadata():
    source_url = request.args.get('url', '').strip()
    if not source_url:
        return jsonify({'error': 'Missing source URL.'}), 400
    try:
        meta = _youtube_metadata(source_url)
        return jsonify(meta)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Return cached chord data for a YouTube URL if available; does not download audio."""
    data = request.get_json(silent=True) or {}
    url = data.get('url', '').strip()

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({'error': 'Invalid YouTube URL.'}), 400

    cached = _cache_get(video_id)
    if cached:
        cached['has_audio'] = False
        cached['source_url'] = url
        job_id = 'cache:' + video_id
        _set_job(job_id, **cached)
        return jsonify({'job_id': job_id, 'cached': True})

    return jsonify({'error': 'No cached chords for this video yet. Upload an MP3 first and include its YouTube link.'}), 404


@app.route('/api/analyze-upload', methods=['POST'])
def analyze_upload():
    file_obj = request.files.get('file')
    if not file_obj or not file_obj.filename:
        return jsonify({'error': 'Please choose an audio file to upload.'}), 400

    filename = file_obj.filename.strip()
    ext = os.path.splitext(filename)[1].lower()
    allowed = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
    if ext and ext not in allowed:
        return jsonify({'error': 'Unsupported file type. Upload mp3, wav, m4a, aac, ogg, or flac.'}), 400

    job_id = str(uuid.uuid4())
    _set_job(job_id, status='processing', progress=0, message='Upload received…')

    source_ext = ext if ext else '.bin'
    source_path = os.path.join(UPLOAD_DIR, f'{job_id}_source{source_ext}')
    file_obj.save(source_path)

    source_url = request.form.get('source_url', '').strip() or request.form.get('youtube_url', '').strip()
    if source_url:
        _set_job(job_id, source_url=source_url)

    title = request.form.get('title', '').strip() or filename
    t = threading.Thread(target=_process_upload_job, args=(job_id, source_path, title), daemon=True)
    t.start()

    return jsonify({'job_id': job_id, 'cached': False})


@app.route('/api/status/<job_id>')
def job_status(job_id):
    return jsonify(_get_job(job_id))


@app.route('/api/health')
def health():
    """Quick diagnostic endpoint."""
    essentia_ok = os.path.exists(_VENV312_PYTHON)
    ffmpeg_ok = shutil.which('ffmpeg') is not None
    return jsonify({
        'essentia_available': essentia_ok,
        'ffmpeg_available': ffmpeg_ok,
        'venv312_path': _VENV312_PYTHON,
        'analyze_script': _ANALYZE_SCRIPT,
        'upload_dir': UPLOAD_DIR,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
