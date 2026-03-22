import Meyda from "meyda";
import { predictChord } from "./predictChord";

// mean of 12-dim vectors
function meanVec(vecs) {
  const m = new Array(12).fill(0);
  for (const v of vecs) for (let i = 0; i < 12; i++) m[i] += v[i];
  for (let i = 0; i < 12; i++) m[i] /= Math.max(1, vecs.length);
  return m;
}

// std of 12-dim vectors
function stdVec(vecs, mean) {
  const s = new Array(12).fill(0);
  for (const v of vecs) for (let i = 0; i < 12; i++) {
    const d = v[i] - mean[i];
    s[i] += d * d;
  }
  for (let i = 0; i < 12; i++) s[i] = Math.sqrt(s[i] / Math.max(1, vecs.length));
  return s;
}

// smooth labels by majority vote over neighbors
function smoothMajority(items, radius = 2) {
  return items.map((_, idx) => {
    const counts = new Map();
    const start = Math.max(0, idx - radius);
    const end = Math.min(items.length - 1, idx + radius);
    for (let j = start; j <= end; j++) {
      const ch = items[j].chord;
      counts.set(ch, (counts.get(ch) || 0) + 1);
    }
    let best = items[idx].chord;
    let bestCt = -1;
    for (const [ch, ct] of counts.entries()) {
      if (ct > bestCt) { bestCt = ct; best = ch; }
    }
    return { ...items[idx], chord: best };
  });
}

// compress timeline (keep only changes)
function compressTimeline(items) {
  const out = [];
  for (const it of items) {
    if (out.length === 0 || out[out.length - 1].chord !== it.chord) out.push(it);
  }
  return out;
}

/**
 * Analyze mono samples and return chord changes.
 * The model expects 24 features per window:
 *   [12 chroma means..., 12 chroma stds...]
 */
export function analyzeChordTimeline(mono, sampleRate, opts = {}) {
  const {
    // frame parameters (for Meyda chroma)
    frameSize = 4096,
    hopSize = 2048,

    // windowing for “one prediction”
    windowSeconds = 1.0,
    windowHopSeconds = 0.5,

    // filtering
    minConfidence = 0.35,

    // smoothing
    smoothRadius = 2,
  } = opts;

  // 1) compute chroma per short frame across entire audio
  const chromaFrames = [];
  for (let i = 0; i + frameSize <= mono.length; i += hopSize) {
    const frame = mono.slice(i, i + frameSize);
    const chroma = Meyda.extract("chroma", frame, { sampleRate, bufferSize: frameSize });
    chromaFrames.push({
      time: i / sampleRate,
      chroma: Array.from(chroma || new Array(12).fill(0)),
    });
  }

  if (chromaFrames.length === 0) return [];

  // 2) aggregate chroma frames into larger windows (mean+std) to match training
  const preds = [];
  const totalSec = mono.length / sampleRate;

  for (let t0 = 0; t0 + windowSeconds <= totalSec; t0 += windowHopSeconds) {
    const t1 = t0 + windowSeconds;
    const framesInWindow = chromaFrames
      .filter(f => f.time >= t0 && f.time < t1)
      .map(f => f.chroma);

    if (framesInWindow.length === 0) continue;

    const m = meanVec(framesInWindow);
    const s = stdVec(framesInWindow, m);
    const feature24 = m.concat(s);

    const { chord, confidence } = predictChord(feature24);
    if (confidence >= minConfidence) {
      preds.push({ time: t0, chord, confidence });
    }
  }

  if (preds.length === 0) return [];

  // 3) smooth and compress
  const smoothed = smoothMajority(preds, smoothRadius);
  const timeline = compressTimeline(smoothed);

  return timeline;
}