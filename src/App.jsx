import { useMemo, useState } from "react";
import { simplifyTimeline } from "./lib/simplifyTimeline";

import { analyzeWithBackend } from "./lib/backendApi";
import { decodeAudioFile, toMono } from "./lib/audio"; // ✅ add back for BPM estimation

import PlayerWithChords from "./components/PlayerWithChords";
import { addBarNumbers } from "./lib/bars";
import { downloadChordSheetPDF } from "./lib/pdf";
import { estimateBPM } from "./lib/tempo";

export default function App() {
  const [file, setFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const [timeline, setTimeline] = useState([]);
  const [timelineWithBars, setTimelineWithBars] = useState([]);
  const [error, setError] = useState("");

  const [easyMode, setEasyMode] = useState(true);
  const [maxChords, setMaxChords] = useState(6);

  const [bpm, setBpm] = useState(120);
  const [beatsPerBar, setBeatsPerBar] = useState(4);
  const [autoBpm, setAutoBpm] = useState(true);

  const displayTimeline = useMemo(() => {
    return easyMode ? simplifyTimeline(timeline, maxChords) : timeline;
  }, [timeline, easyMode, maxChords]);

  async function onAnalyze() {
    if (!file) return;

    setIsRunning(true);
    setError("");
    setTimeline([]);
    setTimelineWithBars([]);

    try {
      // 1) Run backend chord analysis
      const data = await analyzeWithBackend(file);
      const result = data.timeline || [];
      setTimeline(result);

      if (result.length === 0) {
        setError("No confident chords found. Try a different section or file.");
        return;
      }

      // 2) Decide BPM (auto or manual)
      let useBpm = bpm;

      if (autoBpm) {
        try {
          const { audioBuffer } = await decodeAudioFile(file);
          const mono = toMono(audioBuffer);

          const estimated = estimateBPM(mono, audioBuffer.sampleRate);

          // Basic sanity bounds
          if (Number.isFinite(estimated) && estimated >= 40 && estimated <= 240) {
            useBpm = estimated;
            setBpm(estimated);
          }
        } catch {
          // If BPM estimation fails, we keep the existing bpm value silently.
        }
      }

      // 3) Add bar numbers using chosen BPM
      const withBars = addBarNumbers(result, useBpm, beatsPerBar);
      setTimelineWithBars(withBars);
    } catch (e) {
      setError(e?.message || "Failed to analyze audio.");
    } finally {
      setIsRunning(false);
    }
  }

  function onDownloadPDF() {
    if (!timelineWithBars || timelineWithBars.length === 0) return;

    downloadChordSheetPDF({
      title: file ? file.name : "Track2Tabs Chords",
      bpm,
      beatsPerBar,
      timelineWithBars,
    });
  }

  return (
    <div style={{ maxWidth: 900, margin: "20px auto", padding: 0, fontFamily: "system-ui" }}>
      <h1>Track2Tabs 2.0</h1>

      <div
        style={{
          paddingTop: 40,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "flex-start",
        }}
      >
        <h3 style={{ fontSize: 30 }}>Upload</h3>
        <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />

        {file && (
          <p style={{ marginTop: 8 }}>
            <b>Selected File Name:</b> {file.name}
          </p>
        )}

        <button
          style={{ marginTop: 0, padding: "10px 14px", borderRadius: 10, cursor: "pointer" }}
          disabled={!file || isRunning}
          onClick={onAnalyze}
        >
          {isRunning ? "Analyzing..." : "Analyze"}
        </button>

        {error && <p style={{ color: "crimson", marginTop: 10 }}>{error}</p>}
      </div>

      {/* Audio player + live chord pop-up */}
      <PlayerWithChords file={file} timeline={displayTimeline} />

      <div style={{ marginTop: 16, border: "1px solid #ddd", padding: 16, borderRadius: 12 }}>
        <h3 style={{ fontSize: 30 }}>Settings</h3>

        <label style={{ display: "block", marginTop: 10 }}>
          <input type="checkbox" checked={easyMode} onChange={(e) => setEasyMode(e.target.checked)} />{" "}
          Easy mode (simplify progression)
        </label>

        <label style={{ display: "block", marginTop: 10 }}>
          Max chords:
          <input
            type="number"
            min="2"
            max="12"
            value={maxChords}
            onChange={(e) => setMaxChords(parseInt(e.target.value || "6", 10))}
            style={{ marginLeft: 8, width: 70 }}
            disabled={!easyMode}
          />
        </label>

        <hr style={{ margin: "16px 0" }} />

        <label style={{ display: "block", marginTop: 10 }}>
          <input type="checkbox" checked={autoBpm} onChange={(e) => setAutoBpm(e.target.checked)} />{" "}
          Auto-detect BPM (tempo)
        </label>

        <label style={{ display: "block", marginTop: 10 }}>
          BPM:
          <input
            type="number"
            min="40"
            max="240"
            value={bpm}
            onChange={(e) => setBpm(parseInt(e.target.value || "120", 10))}
            style={{ marginLeft: 8, width: 90 }}
            disabled={autoBpm}
          />
        </label>

        <label style={{ display: "block", marginTop: 10 }}>
          Beats per bar:
          <select
            value={beatsPerBar}
            onChange={(e) => setBeatsPerBar(parseInt(e.target.value, 10))}
            style={{ marginLeft: 8 }}
          >
            <option value={3}>3</option>
            <option value={4}>4</option>
            <option value={6}>6</option>
          </select>
        </label>

        <p style={{ marginTop: 10, color: "#555" }}>
          {easyMode
            ? "Easy mode keeps the most common chords and smooths out rare changes."
            : "Easy mode is off, showing the raw predicted chord changes."}
        </p>
      </div>

      <div style={{ marginTop: 16, border: "1px solid #ddd", padding: 16, borderRadius: 12 }}>
        <h3>Chord timeline</h3>

        {displayTimeline.length === 0 ? (
          <p style={{ color: "#555" }}>No results yet.</p>
        ) : (
          <ul>
            {displayTimeline.map((x, idx) => (
              <li key={idx}>
                <b>{x.chord}</b> @ {x.time.toFixed(2)}s{" "}
                {typeof x.confidence === "number" ? `(conf ${x.confidence.toFixed(2)})` : ""}
              </li>
            ))}
          </ul>
        )}

        <button
          style={{ marginTop: 12, padding: "10px 14px", borderRadius: 10, cursor: "pointer" }}
          disabled={!timelineWithBars || timelineWithBars.length === 0}
          onClick={onDownloadPDF}
        >
          Download Chord Sheet (PDF)
        </button>

        {timelineWithBars?.length > 0 && (
          <p style={{ marginTop: 8, color: "#555" }}>
            PDF uses the raw chord timeline + detected/manual BPM to assign bar numbers.
          </p>
        )}
      </div>
    </div>
  );
}