import { useEffect, useMemo, useRef, useState } from "react";

export default function PlayerWithChords({ file, timeline }) {
  const audioRef = useRef(null);
  const rafRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);

  const audioUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  // ✅ NEW: compute active chord using start/end
  const active = useMemo(() => {
    if (!timeline || timeline.length === 0) return null;

    return timeline.find(
      (c) => t >= c.start && t < c.end
    );
  }, [timeline, t]);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  function tick() {
    const el = audioRef.current;
    if (!el) return;

    setT(el.currentTime || 0);
    rafRef.current = requestAnimationFrame(tick);
  }

  function onPlay() {
    setIsPlaying(true);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(tick);
  }

  function onPause() {
    setIsPlaying(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }

  function seek(newT) {
    const el = audioRef.current;
    if (!el) return;

    el.currentTime = newT;
    setT(newT);
  }

  if (!file) return null;

  return (
    <div style={{ border: "1px solid #ddd", padding: 16, borderRadius: 12, marginTop: 16 }}>
      <h3 style={{ fontSize: 30 }}>Player</h3>

      <div style={{ fontSize: 44, fontWeight: 700, margin: "10px 0" }}>
        {active ? active.chord : "—"}
      </div>

      <audio
        ref={audioRef}
        src={audioUrl}
        controls
        onPlay={onPlay}
        onPause={onPause}
        onLoadedMetadata={(e) => setDur(e.currentTarget.duration || 0)}
      />

      <div style={{ marginTop: 10 }}>
        <input
          type="range"
          min={0}
          max={dur || 0}
          step={0.01}
          value={Math.min(t, dur || t)}
          onChange={(e) => seek(parseFloat(e.target.value))}
          style={{ width: "100%" }}
        />

        <div style={{ color: "#555", marginTop: 6 }}>
          {t.toFixed(2)}s / {dur ? dur.toFixed(2) : "?"}s {isPlaying ? "(playing)" : "(paused)"}
        </div>
      </div>
    </div>
  );
}