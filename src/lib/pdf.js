import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

export function downloadChordSheetPDF({ title = "Track2Tabs Chords", bpm, beatsPerBar = 4, timelineWithBars = [] }) {
  const doc = new jsPDF();

  doc.setFontSize(18);
  doc.text(title, 14, 18);

  doc.setFontSize(12);
  doc.text(`BPM: ${bpm}   Time: ${beatsPerBar}/4`, 14, 28);

  const rows = timelineWithBars.map(ev => [
    String(ev.bar ?? ""),
    ev.chord ?? "",
    ev.time != null ? ev.time.toFixed(2) + "s" : "",
    ev.confidence != null ? ev.confidence.toFixed(2) : "",
  ]);

  autoTable(doc, {
    head: [["Bar", "Chord", "Time", "Confidence"]],
    body: rows,
    startY: 36,
  });

  doc.save("track2tabs_chords.pdf");
}