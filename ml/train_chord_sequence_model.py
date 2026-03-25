import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.append(str(BACKEND_DIR))

from chord_model import ChordSequenceModel
from dsp_features import extract_feature_sequence


CHORD_LABELS = [
    "N",
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
]

SR = 22050
FEATURE_DIM = 30
SEQ_LEN = 128
NUM_CLASSES = len(CHORD_LABELS)

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
TRAIN_SIZE = 1000
VAL_SIZE = 200

CPU_COUNT = os.cpu_count() or 8
NUM_PRECOMPUTE_WORKERS = max(2, min(8, CPU_COUNT - 2))

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def setup_torch():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if DEVICE == "cpu":
        torch.set_num_threads(max(1, CPU_COUNT - 1))


def chord_to_index(label):
    return CHORD_LABELS.index(label)


def random_chord_label(rng):
    return rng.choice(CHORD_LABELS[1:])


def chord_tones(label):
    root_map = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    }

    is_minor = label.endswith("m")
    root_name = label[:-1] if is_minor else label

    root = root_map[root_name]
    third = (root + (3 if is_minor else 4)) % 12
    fifth = (root + 7) % 12

    return root, third, fifth


def midi_to_freq(midi):
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synth_note(freq, duration, sr, instrument="sine", amplitude=0.3):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    if instrument == "sine":
        y = np.sin(2 * np.pi * freq * t)
    elif instrument == "piano":
        y = (
            1.0 * np.sin(2 * np.pi * freq * t)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.25 * np.sin(2 * np.pi * 3 * freq * t)
        )
    elif instrument == "guitar":
        y = (
            1.0 * np.sin(2 * np.pi * freq * t)
            + 0.35 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.15 * np.sin(2 * np.pi * 4 * freq * t)
        )
    elif instrument == "pad":
        y = (
            0.8 * np.sin(2 * np.pi * freq * t)
            + 0.6 * np.sin(2 * np.pi * 0.5 * freq * t)
            + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
        )
    else:
        y = np.sin(2 * np.pi * freq * t)

    attack = max(1, int(0.01 * sr))
    release = max(1, int(0.06 * sr))
    env = np.ones_like(y)
    env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    env[-release:] = np.linspace(1, 0, release, dtype=np.float32)

    return (amplitude * y * env).astype(np.float32)


def simple_reverb(y, sr):
    d1 = int(0.03 * sr)
    d2 = int(0.06 * sr)

    out = y.copy()
    if d1 < len(out):
        out[d1:] += 0.25 * y[:-d1]
    if d2 < len(out):
        out[d2:] += 0.12 * y[:-d2]

    return out.astype(np.float32)


def normalize_audio(y):
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 1e-8:
        y = y / peak
    return y.astype(np.float32)


def render_chord_strums(label, duration, sr, rng):
    root, third, fifth = chord_tones(label)

    root_midi_base = rng.choice([36, 48, 60])
    notes = [
        root_midi_base + root,
        root_midi_base + third,
        root_midi_base + fifth,
    ]

    if rng.random() < 0.6:
        notes.append(notes[0] + 12)
    if rng.random() < 0.35:
        notes.append(notes[1] + 12)
    if rng.random() < 0.35:
        inv_idx = rng.choice([0, 1])
        notes[inv_idx] += 12

    instruments = ["guitar", "piano", "pad"]
    chosen = rng.sample(instruments, k=rng.choice([1, 2]))

    total = np.zeros(int(sr * duration), dtype=np.float32)

    # Multiple strums/hits inside one chord region
    num_hits = rng.choice([2, 3, 4, 5])
    hit_times = np.linspace(0, max(0.05, duration - 0.15), num_hits)

    for hit_t in hit_times:
        hit_start = int(hit_t * sr)
        hit_len = len(total) - hit_start
        if hit_len <= 0:
            continue

        for inst in chosen:
            for midi in notes:
                detune_cents = rng.uniform(-8, 8)
                detune_ratio = 2 ** (detune_cents / 1200.0)
                freq = midi_to_freq(midi) * detune_ratio
                amp = rng.uniform(0.08, 0.28)

                note_y = synth_note(freq, duration=0.35, sr=sr, instrument=inst, amplitude=amp)
                note_y = note_y[:hit_len]
                total[hit_start:hit_start + len(note_y)] += note_y

        if rng.random() < 0.7:
            bass_freq = midi_to_freq(notes[0] - 12)
            bass = synth_note(
                bass_freq,
                duration=0.30,
                sr=sr,
                instrument="sine",
                amplitude=rng.uniform(0.04, 0.12),
            )
            bass = bass[:hit_len]
            total[hit_start:hit_start + len(bass)] += bass

    if rng.random() < 0.5:
        total = simple_reverb(total, sr)

    total += np.random.randn(len(total)).astype(np.float32) * rng.uniform(0.001, 0.008)
    return normalize_audio(total)


def render_transition_noise(duration, sr, rng):
    y = np.random.randn(int(sr * duration)).astype(np.float32) * rng.uniform(0.002, 0.02)

    if rng.random() < 0.5:
        fade = np.linspace(1, 0.2, len(y), dtype=np.float32)
        y *= fade

    return normalize_audio(y) * rng.uniform(0.1, 0.4)


def fit_or_crop_features(feats, target_len):
    if len(feats) == 0:
        return np.zeros((target_len, FEATURE_DIM), dtype=np.float32)

    if len(feats) < target_len:
        pad = np.repeat(feats[-1:, :], target_len - len(feats), axis=0)
        return np.vstack([feats, pad]).astype(np.float32)

    if len(feats) > target_len:
        start = 0
        if len(feats) > target_len:
            start = np.random.randint(0, len(feats) - target_len + 1)
        return feats[start:start + target_len].astype(np.float32)

    return feats.astype(np.float32)


def make_realistic_sequence(seed):
    rng = random.Random(seed)
    np.random.seed(seed % (2**32 - 1))

    num_chords = rng.choice([3, 4, 5])
    chosen_labels = [random_chord_label(rng) for _ in range(num_chords)]

    # Build frame allocation with explicit transition/no-chord gaps
    remaining = SEQ_LEN
    chord_frame_lengths = []
    transition_lengths = []

    for i in range(num_chords):
        if i < num_chords - 1:
            trans_len = rng.randint(4, 10)  # transition frames
            transition_lengths.append(trans_len)
            remaining -= trans_len

    for i in range(num_chords):
        if i == num_chords - 1:
            chord_len = remaining
        else:
            min_left = 16 * (num_chords - i - 1)
            chord_len = rng.randint(16, max(16, remaining - min_left))
            remaining -= chord_len
        chord_frame_lengths.append(chord_len)

    all_feats = []
    all_targets = []

    for i, label in enumerate(chosen_labels):
        chord_frames = chord_frame_lengths[i]
        chord_duration = max(0.6, chord_frames * 512 / SR)

        chord_audio = render_chord_strums(label, chord_duration, SR, rng)
        chord_feats = extract_feature_sequence(chord_audio, SR)
        chord_feats = fit_or_crop_features(chord_feats, chord_frames)

        all_feats.append(chord_feats)
        all_targets.extend([chord_to_index(label)] * chord_frames)

        if i < len(transition_lengths):
            trans_frames = transition_lengths[i]
            trans_duration = max(0.15, trans_frames * 512 / SR)

            trans_audio = render_transition_noise(trans_duration, SR, rng)
            trans_feats = extract_feature_sequence(trans_audio, SR)
            trans_feats = fit_or_crop_features(trans_feats, trans_frames)

            all_feats.append(trans_feats)
            all_targets.extend([chord_to_index("N")] * trans_frames)

    full_feats = np.vstack(all_feats).astype(np.float32)
    full_targets = np.asarray(all_targets, dtype=np.int64)

    if len(full_feats) < SEQ_LEN:
        pad_len = SEQ_LEN - len(full_feats)
        full_feats = np.vstack([full_feats, np.repeat(full_feats[-1:, :], pad_len, axis=0)])
        full_targets = np.concatenate([full_targets, np.repeat(full_targets[-1], pad_len)])
    elif len(full_feats) > SEQ_LEN:
        full_feats = full_feats[:SEQ_LEN]
        full_targets = full_targets[:SEQ_LEN]

    return full_feats, full_targets


def build_dataset_parallel(size, base_seed):
    seeds = [base_seed + i for i in range(size)]

    print(f"Precomputing {size} sequences with {NUM_PRECOMPUTE_WORKERS} worker processes...")

    with ProcessPoolExecutor(max_workers=NUM_PRECOMPUTE_WORKERS) as ex:
        items = list(ex.map(make_realistic_sequence, seeds, chunksize=8))

    print(f"Finished precomputing {size} sequences.")
    return items


class RealisticChordDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def save_labels(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chord_labels.json", "w") as f:
        json.dump(CHORD_LABELS, f, indent=2)


def save_checkpoint(model, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": FEATURE_DIM,
            "num_classes": NUM_CLASSES,
            "conv_channels": 64,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
        },
        out_dir / "chord_sequence_model.pt",
    )


def train():
    setup_torch()

    print(f"Using device: {DEVICE}")
    print(f"CPU count: {CPU_COUNT}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Train size: {TRAIN_SIZE}, Val size: {VAL_SIZE}")

    train_items = build_dataset_parallel(TRAIN_SIZE, base_seed=1000)
    val_items = build_dataset_parallel(VAL_SIZE, base_seed=999999)

    train_ds = RealisticChordDataset(train_items)
    val_ds = RealisticChordDataset(val_items)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ChordSequenceModel(
        input_dim=FEATURE_DIM,
        num_classes=NUM_CLASSES,
        conv_channels=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    out_dir = PROJECT_ROOT / "ml" / "artifacts"
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)

            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                yb.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                logits = model(xb)
                loss = criterion(
                    logits.reshape(-1, NUM_CLASSES),
                    yb.reshape(-1),
                )
                val_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch + 1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, out_dir)
            print("Saved new best model.")

    save_labels(out_dir)
    print(f"\nSaved model + labels to: {out_dir}")


if __name__ == "__main__":
    train()