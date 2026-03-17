/* EZChords – frontend logic
 *
 * Flow:
 *  1. User pastes YouTube URL → thumbnail preview shown
 *  2. "Analyze" clicked → POST /api/analyze → job starts
 *  3. Poll /api/status/<jobId> every 2 s until done
 *  4. Render chord items, init YouTube player
 *  5. rAF loop: sync chord track to player current time
 *  6. Beat pulse: advance through beat_times for visual feedback
 */

'use strict';

// ─── State ────────────────────────────────────────────────
let ytPlayer       = null;
let isYTReady      = false;
let pendingVideoId = null;
let audioEl        = null;   // <audio> element used when embedding is blocked
let currentVideoId = null;   // always reflects the video currently loaded in the YT player

let chords     = [];        // [{chord, start, end}, …]
let beatTimes  = [];        // [t0, t1, …] seconds
let bpm        = 120;
let baseKey    = '';        // original detected key
let transposeSteps = 0;     // semitones offset from original
let playbackRate   = 1.0;   // current playback speed

let lyricsLines = [];       // [{text, start, end}, …] from server
let lyricsIdx   = -1;       // index of currently displayed lyric line

let currentIdx = -1;
let beatIdx    = 0;

let rafId      = null;      // requestAnimationFrame handle
let pollTimer  = null;      // setInterval handle for job polling

// Pixels per second of chord duration — sets proportional item widths
const PX_PER_SEC = 130;
const MIN_W      = 160;   // wide enough for 144px diagram + padding
const MAX_W      = 520;

function chordWidth(c) {
  const dur = c.end - c.start;
  return Math.min(MAX_W, Math.max(MIN_W, Math.round(dur * PX_PER_SEC)));
}

function durLabel(c) {
  const d = c.end - c.start;
  return d >= 10 ? `${Math.round(d)}s` : `${d.toFixed(1)}s`;
}

// ─── Utility ──────────────────────────────────────────────

function extractVideoId(url) {
  const pats = [
    /(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/shorts\/)([a-zA-Z0-9_-]{11})/,
    /^([a-zA-Z0-9_-]{11})$/,
  ];
  for (const p of pats) {
    const m = url.match(p);
    if (m) return m[1];
  }
  return null;
}

function formatChord(raw) {
  return raw
    .replace(/([A-G])#/, '$1♯')
    .replace(/([A-G])b/, '$1♭');
}

// ─── Transposition ────────────────────────────────────────
const CHROMATIC = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B'];
// Aliases so both sharp and flat spellings map to the same slot
const NOTE_IDX = {};
CHROMATIC.forEach((n, i) => { NOTE_IDX[n] = i; });
NOTE_IDX['Db'] = 1; NOTE_IDX['D#'] = 3; NOTE_IDX['Gb'] = 6;
NOTE_IDX['G#'] = 8; NOTE_IDX['A#'] = 10;

// Prefer flats for -1 semitone spellings, sharps otherwise
const PREFER_FLAT = new Set(['F','Bb','Eb','Ab','Db']);

function transposeNote(note, steps) {
  const idx = NOTE_IDX[note];
  if (idx === undefined) return note;
  const shifted = ((idx + steps) % 12 + 12) % 12;
  // Pick spelling: use flat if target is one of the flat-key notes
  const sharp = CHROMATIC[shifted];
  // Use flat enharmonic when the sharp has a '#' and flat spelling exists
  const flatMap = {1:'Db',3:'Eb',6:'Gb',8:'Ab',10:'Bb'};
  if (flatMap[shifted] && PREFER_FLAT.has(flatMap[shifted])) return flatMap[shifted];
  return sharp;
}

function transposeChord(chord, steps) {
  if (steps === 0) return chord;
  // Match root (up to 2 chars: letter + optional #/b) then optional suffix
  const m = chord.match(/^([A-G][#b]?)(.*)$/);
  if (!m) return chord;
  return transposeNote(m[1], steps) + m[2];
}

function transposeLabel(steps) {
  if (steps === 0) return 'Original key';
  const abs = Math.abs(steps);
  const dir = steps > 0 ? '▲' : '▼';
  return `${dir} ${abs} semitone${abs > 1 ? 's' : ''}`;
}

const SPEEDS = [0.5, 0.75, 1.0, 1.25, 1.5];

function setSpeed(rate) {
  playbackRate = rate;
  if (audioEl) audioEl.playbackRate = rate;
  if (ytPlayer && ytPlayer.setPlaybackRate) ytPlayer.setPlaybackRate(rate);
  document.getElementById('speedLabel').textContent =
    rate === 1 ? 'Normal speed' : `Speed: ${rate}×`;
}

function applyTranspose(steps) {
  transposeSteps = steps;
  document.getElementById('transposeLabel').textContent = transposeLabel(steps);
  if (baseKey) {
    document.getElementById('keyBadge').textContent = `Key: ${transposeChord(baseKey, steps)}`;
  }
  // Re-label chord names and diagrams
  document.querySelectorAll('#chordRow .chord-item').forEach((el, i) => {
    const transposed = transposeChord(chords[i].chord, steps);
    const nameEl = el.querySelector('.chord-name');
    if (nameEl) nameEl.textContent = formatChord(transposed);
    const diagEl = el.querySelector('.chord-diagram');
    if (diagEl) diagEl.innerHTML = buildChordSVG(transposed);
  });
}

// ─── Chord Diagrams ───────────────────────────────────────
// strings: [s6..s1] relative fret positions (1-4), 0=open, -1=mute
// b: baseFret (actual fret number of display row 1)
const CHORD_DIAGRAMS = {
  'C':   { f:[-1,3,2,0,1,0],     b:1 },
  'C#':  { f:[-1,1,3,3,3,1],     b:4 },
  'D':   { f:[-1,-1,0,2,3,2],    b:1 },
  'Eb':  { f:[-1,1,3,3,3,1],     b:6 },
  'E':   { f:[0,2,2,1,0,0],      b:1 },
  'F':   { f:[1,3,3,2,1,1],      b:1 },
  'F#':  { f:[1,3,3,2,1,1],      b:2 },
  'G':   { f:[3,2,0,0,0,3],      b:1 },
  'Ab':  { f:[1,3,3,2,1,1],      b:4 },
  'A':   { f:[-1,0,2,2,2,0],     b:1 },
  'Bb':  { f:[-1,1,3,3,3,1],     b:1 },
  'B':   { f:[-1,1,3,3,3,1],     b:2 },
  'Cm':  { f:[-1,1,3,3,2,1],     b:3 },
  'C#m': { f:[-1,1,3,3,2,1],     b:4 },
  'Dm':  { f:[-1,-1,0,2,3,1],    b:1 },
  'Ebm': { f:[-1,1,3,3,2,1],     b:6 },
  'Em':  { f:[0,2,2,0,0,0],      b:1 },
  'Fm':  { f:[1,3,3,1,1,1],      b:1 },
  'F#m': { f:[1,3,3,1,1,1],      b:2 },
  'Gm':  { f:[1,3,3,1,1,1],      b:3 },
  'Abm': { f:[1,3,3,1,1,1],      b:4 },
  'Am':  { f:[-1,0,2,2,1,0],     b:1 },
  'Bbm': { f:[-1,1,3,3,2,1],     b:1 },
  'Bm':  { f:[-1,1,3,3,2,1],     b:2 },
  // aliases for transposed spellings
  'Db':  { f:[-1,1,3,3,3,1],     b:4 },
  'Gb':  { f:[1,3,3,2,1,1],      b:2 },
  'Dbm': { f:[-1,1,3,3,2,1],     b:4 },
  'Gbm': { f:[1,3,3,1,1,1],      b:2 },
};

function buildChordSVG(chordName) {
  const key  = chordName.replace('♯','#').replace('♭','b');
  const data = CHORD_DIAGRAMS[key];
  if (!data) return '';
  const { f: strings, b: baseFret } = data;

  const SX       = [6, 17, 28, 39, 50, 61];  // string x positions (s6..s1)
  const NUT_Y    = 16;
  const FRET_H   = 12;
  const N_FRETS  = 4;
  const DOT_R    = 4.5;
  const bottom   = NUT_Y + N_FRETS * FRET_H;  // 64

  const p = [];

  // Nut or position marker
  if (baseFret === 1) {
    p.push(`<rect x="${SX[0]}" y="${NUT_Y-3}" width="${SX[5]-SX[0]}" height="3" fill="currentColor" opacity=".6"/>`);
  } else {
    p.push(`<line x1="${SX[0]}" y1="${NUT_Y}" x2="${SX[5]}" y2="${NUT_Y}" stroke="currentColor" stroke-width="1.2" opacity=".3"/>`);
    // fret number: floats to the left of the leftmost string, no background
    const lblY = NUT_Y + FRET_H * 0.75;
    p.push(`<text x="-2" y="${lblY}" text-anchor="middle" font-size="8" font-weight="700" font-family="sans-serif" fill="currentColor" opacity=".85">${baseFret}</text>`);
  }

  // Fret lines
  for (let i = 1; i <= N_FRETS; i++) {
    const y = NUT_Y + i * FRET_H;
    p.push(`<line x1="${SX[0]}" y1="${y}" x2="${SX[5]}" y2="${y}" stroke="currentColor" stroke-width=".8" opacity=".18"/>`);
  }

  // String lines
  for (const x of SX) {
    p.push(`<line x1="${x}" y1="${NUT_Y}" x2="${x}" y2="${bottom}" stroke="currentColor" stroke-width="1" opacity=".28"/>`);
  }

  // Open / muted indicators above nut
  for (let i = 0; i < 6; i++) {
    const x = SX[i], fret = strings[i];
    if (fret === -1) {
      p.push(`<text x="${x}" y="12" text-anchor="middle" font-size="10" font-family="sans-serif" fill="currentColor" opacity=".5">×</text>`);
    } else if (fret === 0) {
      p.push(`<circle cx="${x}" cy="8" r="3.5" fill="none" stroke="currentColor" stroke-width="1" opacity=".45"/>`);
    }
  }

  // Finger dots
  for (let i = 0; i < 6; i++) {
    const fret = strings[i];
    if (fret > 0) {
      const x = SX[i];
      const y = NUT_Y + (fret - 0.5) * FRET_H;
      p.push(`<circle cx="${x}" cy="${y}" r="${DOT_R}" fill="currentColor" opacity=".88"/>`);
    }
  }

  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-8 0 84 72" width="144" height="154" aria-hidden="true">${p.join('')}</svg>`;
}

function showOnly(sectionId) {
  ['inputSection', 'loadingSection', 'errorSection', 'resultsSection'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden', id !== sectionId);
  });
}

// ─── YouTube IFrame API ───────────────────────────────────

// Called automatically by the YouTube script when it's ready
window.onYouTubeIframeAPIReady = function () {
  isYTReady = true;
  if (pendingVideoId) {
    createPlayer(pendingVideoId);
    pendingVideoId = null;
  }
};

function createPlayer(videoId) {
  currentVideoId = videoId;   // update before any async events can fire
  _hideAudioPlayer();
  if (ytPlayer) {
    ytPlayer.loadVideoById(videoId);
    return;
  }
  ytPlayer = new YT.Player('yt-player', {
    width: '100%',
    videoId,
    playerVars: { autoplay: 1, controls: 1, rel: 0, origin: window.location.origin },
    events: {
      onReady: (e) => { e.target.playVideo(); startTracking(); },
      onStateChange: (e) => {
        if (e.data === YT.PlayerState.PLAYING) startTracking();
        if (e.data === YT.PlayerState.PAUSED || e.data === YT.PlayerState.ENDED) {
          stopTracking();
        }
      },
      onError: (e) => {
        // 101 & 150 = embedding disabled by video owner
        // Use currentVideoId (not closure var) so subsequent loadVideoById calls work correctly
        if (e.data === 100 || e.data === 101 || e.data === 150) {
          _showAudioPlayer(currentVideoId);
        }
      },
    },
  });
}

function _showAudioPlayer(videoId) {
  document.getElementById('playerWrapper').classList.add('hidden');
  const panel = document.getElementById('audioPlayerPanel');
  panel.classList.remove('hidden');
  document.getElementById('apYtLink').href = `https://www.youtube.com/watch?v=${videoId}`;

  audioEl = document.getElementById('audioEl');
  audioEl.src = `/api/audio/${videoId}`;

  audioEl.addEventListener('play',  () => startTracking(), { once: false });
  audioEl.addEventListener('pause', () => stopTracking(),  { once: false });
  audioEl.addEventListener('ended', () => stopTracking(),  { once: false });
  audioEl.addEventListener('seeked', () => { currentIdx = -1; beatIdx = 0; }, { once: false });

  audioEl.play().catch(() => {});
}

function _hideAudioPlayer() {
  audioEl = null;
  document.getElementById('playerWrapper').classList.remove('hidden');
  document.getElementById('audioPlayerPanel').classList.add('hidden');
  const el = document.getElementById('audioEl');
  el.pause();
  el.src = '';
}

// ─── Chord Rendering ──────────────────────────────────────

function renderChordRow(chordsData) {
  const row = document.getElementById('chordRow');
  row.innerHTML = '';
  chordsData.forEach((c, i) => {
    const div = document.createElement('div');
    div.className = 'chord-item';
    const w = chordWidth(c);
    div.style.width = w + 'px';
    div.dataset.idx = i;

    const transposed = transposeChord(c.chord, transposeSteps);

    const name = document.createElement('span');
    name.className   = 'chord-name';
    name.textContent = formatChord(transposed);

    const diag = document.createElement('span');
    diag.className = 'chord-diagram';
    diag.innerHTML = buildChordSVG(transposed);

    const dur = document.createElement('span');
    dur.className   = 'chord-dur';
    dur.textContent = durLabel(c);

    div.appendChild(name);
    div.appendChild(diag);
    div.appendChild(dur);

    div.addEventListener('click', () => {
      if (audioEl) {
        audioEl.currentTime = c.start;
        audioEl.play();
      } else if (ytPlayer && ytPlayer.seekTo) {
        ytPlayer.seekTo(c.start, true);
        ytPlayer.playVideo();
      }
    });
    row.appendChild(div);
  });
}

// ─── Chord Track Animation ────────────────────────────────

function centerOnIndex(idx) {
  const row      = document.getElementById('chordRow');
  const viewport = document.getElementById('chordViewport');
  const vpW      = viewport.clientWidth;
  const items    = row.querySelectorAll('.chord-item');
  if (!items[idx]) return;
  const item   = items[idx];
  let offset = -(item.offsetLeft + item.offsetWidth / 2 - vpW / 2);

  // Always keep at least 80px of the next chord peeking in from the right
  const PEEK = 80;
  if (items[idx + 1]) {
    const nextItem  = items[idx + 1];
    const maxOffset = vpW - nextItem.offsetLeft - PEEK;
    offset = Math.min(offset, maxOffset);
  }

  row.style.transform = `translateX(${offset}px)`;
}

function applyDistanceClasses(centerIdx) {
  const items = document.querySelectorAll('#chordRow .chord-item');
  items.forEach((item, i) => {
    const dist = i - centerIdx;
    const abs  = Math.abs(dist);
    item.classList.remove('active', 'near-1', 'near-2', 'entering', 'beat');
    if      (abs === 0) item.classList.add('active');
    else if (abs === 1) item.classList.add('near-1');
    else if (abs === 2) item.classList.add('near-2');
  });
}

function triggerEnterAnimation() {
  const active = document.querySelector('#chordRow .chord-item.active');
  if (!active) return;
  active.classList.remove('entering');
  void active.offsetWidth; // force reflow
  active.classList.add('entering');
}

function triggerBeatPulse() {
  const active = document.querySelector('#chordRow .chord-item.active');
  if (!active) return;
  active.classList.remove('beat');
  void active.offsetWidth;
  active.classList.add('beat');
}

// ─── Lyrics ───────────────────────────────────────────────

function findLyricAt(t) {
  // Return index of the lyric cue active at time t, or last past cue
  for (let i = 0; i < lyricsLines.length; i++) {
    if (lyricsLines[i].start <= t && lyricsLines[i].end > t) return i;
  }
  for (let i = lyricsLines.length - 1; i >= 0; i--) {
    if (lyricsLines[i].start <= t) return i;
  }
  return -1;
}

function updateLyrics(t) {
  if (!lyricsLines.length) return;
  const idx = findLyricAt(t);
  if (idx === lyricsIdx) return;
  lyricsIdx = idx;

  const prevEl   = document.getElementById('lyricPrev');
  const activeEl = document.getElementById('lyricActive');
  const nextEl   = document.getElementById('lyricNext');

  const newPrev   = idx > 0 ? lyricsLines[idx - 1].text : '';
  const newActive = idx >= 0 ? lyricsLines[idx].text : '♪';
  const newNext   = idx >= 0 && idx < lyricsLines.length - 1 ? lyricsLines[idx + 1].text : '';

  // Animate each line out then in with new content
  function animateLine(el, newText) {
    el.classList.remove('lyric-entering', 'lyric-exiting');
    void el.offsetWidth; // force reflow
    el.classList.add('lyric-exiting');
    const onEnd = () => {
      el.removeEventListener('animationend', onEnd);
      el.textContent = newText;
      el.classList.remove('lyric-exiting');
      void el.offsetWidth;
      el.classList.add('lyric-entering');
    };
    el.addEventListener('animationend', onEnd);
  }

  animateLine(prevEl,   newPrev);
  animateLine(activeEl, newActive);
  animateLine(nextEl,   newNext);
}

// ─── rAF Tracking Loop ────────────────────────────────────

function findChordAt(t) {
  // Binary search
  let lo = 0, hi = chords.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if      (chords[mid].end   <= t) lo = mid + 1;
    else if (chords[mid].start >  t) hi = mid - 1;
    else return mid;
  }
  return -1;
}

function trackLoop() {
  let t;
  if (audioEl && !audioEl.paused) {
    t = audioEl.currentTime;
  } else if (ytPlayer && typeof ytPlayer.getCurrentTime === 'function') {
    t = ytPlayer.getCurrentTime();
  } else {
    rafId = requestAnimationFrame(trackLoop);
    return;
  }

  const idx = findChordAt(t);

  // ── Beat pulse ──
  while (beatIdx < beatTimes.length && beatTimes[beatIdx] <= t) {
    triggerBeatPulse();
    beatIdx++;
  }

  // ── Lyrics ──
  updateLyrics(t);

  // ── Active-chord fill (sweeps left→right inside the chord box) ──
  if (idx >= 0) {
    const chord    = chords[idx];
    const elapsed  = t - chord.start;
    const duration = chord.end - chord.start;
    const fill     = Math.min(1, elapsed / duration);
    const activeEl = document.querySelector('#chordRow .chord-item.active');
    if (activeEl) activeEl.style.setProperty('--fill', fill);
  }

  // ── Chord changed ──
  if (idx !== currentIdx) {
    currentIdx = idx;
    if (idx >= 0) {
      applyDistanceClasses(idx);
      // Reset fill on new active chord so sweep starts from zero
      const newActive = document.querySelector('#chordRow .chord-item.active');
      if (newActive) newActive.style.setProperty('--fill', '0');
      centerOnIndex(idx);
      triggerEnterAnimation();
    }
  }

  rafId = requestAnimationFrame(trackLoop);
}

function startTracking() {
  stopTracking();
  beatIdx = 0;
  // Fast-forward beatIdx to current time
  if (ytPlayer && ytPlayer.getCurrentTime) {
    const t = ytPlayer.getCurrentTime();
    while (beatIdx < beatTimes.length && beatTimes[beatIdx] < t) beatIdx++;
  }
  rafId = requestAnimationFrame(trackLoop);
}

function stopTracking() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}

// ─── Job Polling ──────────────────────────────────────────

function startPolling(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => pollJob(jobId), 2000);
  pollJob(jobId); // immediate first check
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

async function pollJob(jobId) {
  try {
    const res  = await fetch(`/api/status/${jobId}`);
    const data = await res.json();

    if (data.status === 'processing') {
      document.getElementById('loadingMessage').textContent = data.message  || 'Processing…';
      document.getElementById('loadingProgress').style.width = (data.progress || 0) + '%';
      return;
    }

    stopPolling();

    if (data.status === 'done') {
      loadResults(data);
    } else {
      showError(data.message || 'Unknown error during analysis.');
    }
  } catch (err) {
    // network hiccup – keep polling
  }
}

// ─── Results Display ──────────────────────────────────────

let _lyricsPollTimer = null;

function _applyLyrics(lines) {
  lyricsLines = lines;
  lyricsIdx   = -1;
  const lyricsSection = document.getElementById('lyricsSection');
  if (lyricsLines.length) {
    document.getElementById('lyricPrev').textContent   = '';
    document.getElementById('lyricActive').textContent = '♪';
    document.getElementById('lyricNext').textContent   = lyricsLines[0] ? lyricsLines[0].text : '';
    lyricsSection.classList.remove('hidden');
  } else {
    lyricsSection.classList.add('hidden');
  }
}

async function _pollLyrics(videoId, attempts = 0) {
  if (attempts > 12) return; // stop after ~60 s
  try {
    const res  = await fetch(`/api/lyrics/${videoId}`);
    const json = await res.json();
    if (json.lyrics && json.lyrics.length) {
      _applyLyrics(json.lyrics);
      return;
    }
    if (json.status === 'fetching' || json.status === 'ok') {
      _lyricsPollTimer = setTimeout(() => _pollLyrics(videoId, attempts + 1), 5000);
    }
  } catch (_) {}
}

function loadResults(data) {
  chords    = data.chords    || [];
  beatTimes = data.beat_times|| [];
  bpm       = data.bpm       || 120;
  baseKey   = data.key       || '';
  transposeSteps = 0;   // reset to original key for each new song
  playbackRate   = 1.0; // reset speed for each new song
  currentIdx = -1;
  beatIdx    = 0;

  // Clear any previous lyrics poll
  if (_lyricsPollTimer) { clearTimeout(_lyricsPollTimer); _lyricsPollTimer = null; }

  // Apply lyrics from job data if present, else fetch from API
  if (data.lyrics && data.lyrics.length) {
    _applyLyrics(data.lyrics);
  } else if (data.video_id) {
    _applyLyrics([]);
    _pollLyrics(data.video_id);
  } else {
    _applyLyrics([]);
  }

  document.getElementById('songTitle').textContent   = data.title || '—';
  document.getElementById('keyBadge').textContent    = `Key: ${data.key || '—'}`;
  document.getElementById('bpmBadge').textContent    = `BPM: ${data.bpm || '—'}`;
  document.getElementById('transposeLabel').textContent = 'Original key';
  setSpeed(1.0);

  renderChordRow(chords);

  showOnly('resultsSection');

  // Init or reload the player
  if (isYTReady) {
    createPlayer(data.video_id);
  } else {
    pendingVideoId = data.video_id;
  }
}

// ─── UI Helpers ───────────────────────────────────────────

function showError(msg) {
  document.getElementById('errorMessage').textContent = msg;
  showOnly('errorSection');
}

function resetToInput() {
  stopTracking();
  stopPolling();
  if (_lyricsPollTimer) { clearTimeout(_lyricsPollTimer); _lyricsPollTimer = null; }
  if (ytPlayer) { try { ytPlayer.stopVideo(); } catch (_) {} }
  _hideAudioPlayer();
  chords = []; beatTimes = []; currentIdx = -1;
  lyricsLines = []; lyricsIdx = -1;
  document.getElementById('lyricsSection').classList.add('hidden');
  showOnly('inputSection');
  loadCachedList(); // refresh in case a new song was just analyzed
}

// ─── Boot ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {

  const urlInput   = document.getElementById('youtubeUrl');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const thumbPrev  = document.getElementById('thumbPreview');
  const thumbImg   = document.getElementById('thumbImg');

  // Live thumbnail preview
  urlInput.addEventListener('input', () => {
    const vid = extractVideoId(urlInput.value.trim());
    if (vid) {
      thumbImg.src = `https://img.youtube.com/vi/${vid}/mqdefault.jpg`;
      thumbPrev.classList.remove('hidden');
    } else {
      thumbPrev.classList.add('hidden');
    }
  });

  // Analyze button
  analyzeBtn.addEventListener('click', submitUrl);
  urlInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') submitUrl(); });

  document.getElementById('retryBtn').addEventListener('click', resetToInput);
  document.getElementById('newSongBtn').addEventListener('click', resetToInput);

  document.getElementById('transposeDown').addEventListener('click', () => applyTranspose(transposeSteps - 1));
  document.getElementById('transposeUp').addEventListener('click',   () => applyTranspose(transposeSteps + 1));

  document.getElementById('speedDown').addEventListener('click', () => {
    const idx = SPEEDS.indexOf(playbackRate);
    if (idx > 0) setSpeed(SPEEDS[idx - 1]);
  });
  document.getElementById('speedUp').addEventListener('click', () => {
    const idx = SPEEDS.indexOf(playbackRate);
    if (idx < SPEEDS.length - 1) setSpeed(SPEEDS[idx + 1]);
  });

  // Load cached songs list on page load
  loadCachedList();
});

async function submitUrl() {
  const url = document.getElementById('youtubeUrl').value.trim();
  if (!url) return;

  if (!extractVideoId(url)) {
    showError('That doesn\'t look like a valid YouTube URL. Please try again.');
    return;
  }

  showOnly('loadingSection');
  document.getElementById('loadingMessage').textContent = 'Starting…';
  document.getElementById('loadingProgress').style.width = '0%';

  try {
    const res  = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    if (data.error) { showError(data.error); return; }

    if (data.cached) {
      // Already analyzed — fetch result directly
      const res2  = await fetch(`/api/status/${data.job_id}`);
      const data2 = await res2.json();
      if (data2.status === 'done') { loadResults(data2); return; }
    }

    startPolling(data.job_id);
  } catch (err) {
    showError('Could not reach the server. Is the Flask app running?');
  }
}

// ─── Cached Songs Panel ───────────────────────────────────

async function loadCachedList() {
  try {
    const res   = await fetch('/api/cached');
    const songs = await res.json();
    renderCachedList(songs);
  } catch (_) { /* server not available yet */ }
}

function renderCachedList(songs) {
  const section = document.getElementById('cachedSection');
  const list    = document.getElementById('cachedList');
  if (!songs || songs.length === 0) {
    section.classList.add('hidden');
    return;
  }

  section.classList.remove('hidden');
  list.innerHTML = '';

  songs.forEach(song => {
    const item = document.createElement('div');
    item.className = 'cached-item';

    const thumb = document.createElement('img');
    thumb.src = `https://img.youtube.com/vi/${song.video_id}/mqdefault.jpg`;
    thumb.alt = song.title;
    thumb.loading = 'lazy';

    const meta = document.createElement('div');
    meta.className = 'cached-item-meta';
    meta.innerHTML = `
      <div class="cached-item-title" title="${escHtml(song.title)}">${escHtml(song.title)}</div>
      <div class="cached-item-sub">Key: ${escHtml(song.key)} &nbsp;·&nbsp; ${song.bpm} BPM</div>
    `;

    const del = document.createElement('button');
    del.className  = 'cached-item-del';
    del.title = 'Remove from cache';
    del.textContent = '✕';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      await fetch(`/api/cached/${song.video_id}`, { method: 'DELETE' });
      item.remove();
      if (list.children.length === 0) {
        document.getElementById('cachedSection').classList.add('hidden');
      }
    });

    item.appendChild(thumb);
    item.appendChild(meta);
    item.appendChild(del);

    item.addEventListener('click', () => loadCachedSong(song.video_id));
    list.appendChild(item);
  });
}

async function loadCachedSong(videoId) {
  showOnly('loadingSection');
  document.getElementById('loadingMessage').textContent = 'Loading from cache…';
  document.getElementById('loadingProgress').style.width = '80%';

  try {
    const res  = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: `https://www.youtube.com/watch?v=${videoId}` }),
    });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }

    const res2  = await fetch(`/api/status/${data.job_id}`);
    const data2 = await res2.json();
    if (data2.status === 'done') { loadResults(data2); }
    else { startPolling(data.job_id); }
  } catch (err) {
    showError('Could not reach the server.');
  }
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
