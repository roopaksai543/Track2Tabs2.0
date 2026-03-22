export function addBarNumbers(timeline, bpm, beatsPerBar = 4) {
  const secPerBeat = 60 / bpm;
  const secPerBar = beatsPerBar * secPerBeat;

  return (timeline || []).map(ev => ({
    ...ev,
    bar: Math.floor(ev.time / secPerBar) + 1,
  }));
}