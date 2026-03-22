import model from "../assets/chord_model.json";

// dot product: sum(feature[i] * weight[i])
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// turns scores into probabilities
function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - m));
  const sum = exps.reduce((p, c) => p + c, 0);
  return exps.map(e => e / sum);
}

/**
 * feature24 = [12 chroma means..., 12 chroma stds...]
 */
export function predictChord(feature24) {
  const { labels, coef, intercept } = model;

  // score each chord class: score_k = W_k dot x + b_k
  const logits = coef.map((row, k) => dot(row, feature24) + intercept[k]);

  const probs = softmax(logits);

  // pick best chord
  let best = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[best]) best = i;
  }

  return {
    chord: labels[best],
    confidence: probs[best],
  };
}