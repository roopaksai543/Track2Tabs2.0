import json
from pathlib import Path

import numpy as np
import torch

from chord_model import ChordSequenceModel
from dsp_features import load_audio, extract_feature_sequence, get_frame_times


class ChordInferenceEngine:
    def __init__(self, model_path, labels_path, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        model_path = Path(model_path)
        labels_path = Path(labels_path)

        with open(labels_path, "r") as f:
            self.labels = json.load(f)

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = ChordSequenceModel(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            conv_channels=checkpoint.get("conv_channels", 64),
            hidden_size=checkpoint.get("hidden_size", 128),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=checkpoint.get("dropout", 0.2),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict_file(self, audio_path):
        y, sr = load_audio(audio_path)
        feats = extract_feature_sequence(y, sr)

        if len(feats) == 0:
            return []

        times = get_frame_times(len(feats), sr)

        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = np.argmax(probs, axis=-1)
        conf = np.max(probs, axis=-1)

        labels = [self.labels[i] for i in pred_idx]

        # Stronger smoothing than before
        labels = median_vote_smooth(labels, window_size=11)
        labels = smooth_labels(labels, min_run=18)

        timeline = compress_timeline(times, labels, conf)
        timeline = merge_short_segments(timeline, min_duration=0.75)
        timeline = merge_same_label_neighbors(timeline)

        return timeline


def median_vote_smooth(labels, window_size=11):
    if not labels:
        return []

    half = window_size // 2
    out = []

    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        window = labels[lo:hi]

        counts = {}
        for label in window:
            counts[label] = counts.get(label, 0) + 1

        best = max(counts.items(), key=lambda x: x[1])[0]
        out.append(best)

    return out


def smooth_labels(labels, min_run=18):
    if not labels:
        return labels[:]

    out = labels[:]
    n = len(out)
    start = 0

    while start < n:
        end = start + 1
        while end < n and out[end] == out[start]:
            end += 1

        run_len = end - start

        if run_len < min_run:
            left = out[start - 1] if start > 0 else None
            right = out[end] if end < n else None

            if left is not None and right is not None:
                replacement = left if left == right else left
            else:
                replacement = left or right or out[start]

            for i in range(start, end):
                out[i] = replacement

        start = end

    return out


def compress_timeline(times, labels, conf):
    if len(times) == 0 or len(labels) == 0:
        return []

    timeline = []
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segment_conf = float(np.mean(conf[start_idx:i])) if i > start_idx else float(conf[i - 1])
            timeline.append(
                {
                    "start": float(times[start_idx]),
                    "end": float(times[i]),
                    "chord": labels[i - 1],
                    "confidence": segment_conf,
                }
            )
            start_idx = i

    segment_conf = float(np.mean(conf[start_idx:])) if len(conf[start_idx:]) > 0 else 0.0
    timeline.append(
        {
            "start": float(times[start_idx]),
            "end": float(times[-1]),
            "chord": labels[-1],
            "confidence": segment_conf,
        }
    )

    return timeline


def merge_short_segments(timeline, min_duration=0.75):
    if not timeline:
        return []

    out = [timeline[0]]

    for seg in timeline[1:]:
        prev = out[-1]
        duration = seg["end"] - seg["start"]

        if duration < min_duration:
            prev["end"] = seg["end"]
            prev["confidence"] = max(prev.get("confidence", 0.0), seg.get("confidence", 0.0))
        else:
            out.append(seg)

    return out


def merge_same_label_neighbors(timeline):
    if not timeline:
        return []

    merged = [timeline[0]]

    for seg in timeline[1:]:
        prev = merged[-1]

        if seg["chord"] == prev["chord"]:
            prev["end"] = seg["end"]
            prev["confidence"] = max(prev.get("confidence", 0.0), seg.get("confidence", 0.0))
        else:
            merged.append(seg)

    return merged