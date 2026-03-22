export function getActiveChord(timeline, currentTimeSec) {
  if (!timeline || timeline.length === 0) return null;

  // timeline is sorted by time
  let active = timeline[0];
  for (let i = 1; i < timeline.length; i++) {
    if (timeline[i].time <= currentTimeSec) active = timeline[i];
    else break;
  }
  return active;
}