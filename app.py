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
from scipy.ndimage import median_filter

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
# Chord templates (major + minor, 24 total)
# ─────────────────────────────────────────────
NOTES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _build_templates():
    templates = {}
    for i, note in enumerate(NOTES):
        for chord_type, intervals in [('', [0, 4, 7]), ('m', [0, 3, 7])]:
            t = np.zeros(12)
            for iv in intervals:
                t[(i + iv) % 12] = 1.0
            t /= np.linalg.norm(t)
            templates[note + chord_type] = t
    return templates


CHORD_TEMPLATES = _build_templates()
ALL_CHORDS = list(CHORD_TEMPLATES.keys())
TEMPLATE_MATRIX = np.array([CHORD_TEMPLATES[c] for c in ALL_CHORDS])  # (24, 12)

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

def detect_chords(audio_path: str, hop_size: float = 0.5):
    """Return (chords_list, bpm, key_string, beat_times_list)."""
    y, sr = librosa.load(audio_path, mono=True, sr=22050, duration=360)

    hop_length = int(hop_size * sr)

    # Separate harmonic / percussive
    y_harm, y_perc = librosa.effects.hpss(y)

    # Chroma from harmonic component
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, hop_length=hop_length,
        bins_per_octave=24, n_chroma=12,
    )

    # Temporal smoothing
    chroma_smooth = median_filter(chroma, size=(1, 9))

    # Beat tracking from percussive component
    # librosa ≥0.10 returns tempo as a 1-element array; .item() handles both scalar and array
    tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr, hop_length=hop_length)
    tempo = np.asarray(tempo).item()
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length).tolist()

    # Detect tonal key (most prominent pitch class)
    key_idx = int(np.argmax(np.mean(chroma_smooth, axis=1)))
    key = NOTES[key_idx]

    # Per-frame chord matching
    raw = []
    n_frames = chroma_smooth.shape[1]
    for i in range(n_frames):
        vec = chroma_smooth[:, i]
        s = vec.sum()
        if s > 0:
            vec = vec / s
        idx = int(np.argmax(TEMPLATE_MATRIX @ vec))
        raw.append({
            'chord': ALL_CHORDS[idx],
            'start': round(i * hop_size, 3),
            'end': round((i + 1) * hop_size, 3),
        })

    # Merge consecutive identical chords
    merged = []
    for c in raw:
        if merged and merged[-1]['chord'] == c['chord']:
            merged[-1]['end'] = c['end']
        else:
            merged.append(dict(c))

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
        _set_job(job_id, status='error', message=str(exc))
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
