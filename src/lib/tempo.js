export function estimateBPM(mono, sr) {
  // 1) compute short-time energy envelope
  const hop = 512;
  const win = 1024;
  const env = [];
  for (let i = 0; i + win < mono.length; i += hop) {
    let s = 0;
    for (let j = 0; j < win; j++) {
      const x = mono[i + j];
      s += x * x;
    }
    env.push(s);
  }

  // 2) remove DC (mean)
  const mean = env.reduce((a, b) => a + b, 0) / Math.max(1, env.length);
  const x = env.map(v => v - mean);

  // 3) autocorrelation to find periodicity
  // Search BPM range ~ 70–180
  const minBPM = 70;
  const maxBPM = 180;
  const secPerFrame = hop / sr;

  const minLag = Math.floor((60 / maxBPM) / secPerFrame);
  const maxLag = Math.floor((60 / minBPM) / secPerFrame);

  let bestLag = minLag;
  let best = -Infinity;

  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i + lag < x.length; i++) sum += x[i] * x[i + lag];
    if (sum > best) { best = sum; bestLag = lag; }
  }

  const periodSec = bestLag * secPerFrame;
  const bpm = 60 / periodSec;
  return Math.round(bpm);
}