function normalizeChord(ch) {
  // keep only major/minor triads from labels like "C" or "Cm"
  // (your model already outputs these, so this is mostly future-proof)
  return ch;
}

export function simplifyTimeline(timeline, maxChords = 6) {
  if (!timeline || timeline.length === 0) return [];

  // Count how long each chord lasts (using time gaps)
  const durations = new Map();
  for (let i = 0; i < timeline.length; i++) {
    const cur = normalizeChord(timeline[i].chord);
    const t0 = timeline[i].time;
    const t1 = (i + 1 < timeline.length) ? timeline[i + 1].time : (t0 + 1.0);
    const dt = Math.max(0.01, t1 - t0);
    durations.set(cur, (durations.get(cur) || 0) + dt);
  }

  // Keep top N chords by total duration
  const top = [...durations.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxChords)
    .map(x => x[0]);

  const allowed = new Set(top);

  // Replace rare chords with the nearest previous allowed chord
  const out = [];
  let last = null;
  for (const item of timeline) {
    const ch = normalizeChord(item.chord);
    const use = allowed.has(ch) ? ch : last;
    if (!use) continue;
    if (out.length === 0 || out[out.length - 1].chord !== use) {
      out.push({ ...item, chord: use });
    }
    last = use;
  }
  return out;
}