import json
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# CONFIG (EDIT THESE ONLY)
# =========================
SEED = 10 # (0 - 10000)

# Dataset size (bigger = slower)
SAMPLES_PER_CLASS = 4000 # How many samples are generated for each chord
TEST_SIZE = 0.15 # The percentage of samples that are tested (20% test, 80% train)
RANDOM_STATE = 10 # Think of it as a shuffling pattern (0 - 10000)

# Audio synthesis
SR = 22050 # Sample rate (number of samples per second)
DUR = 1.0 # The duration of the synthetic chord sample
BASE_MIDI_CHOICES = [40, 45, 48, 52, 55, 60] # MIDI note numbers used to represent octaves to test
HARMONICS_CHOICES = [4, 6, 8] # Represents overtones to make audio realistic
DETUNE_CENTS_RANGE = (-18, 18) # Slight pitch up or down range of chords
AMP_RANGE = (0.35, 1.0) # The range of the amplitude
NOISE_STD = 0.01 # The amount of noise that gets added 

# Envelope
ATTACK_FRAC = 0.02 # Audio attack percentage (fade in)
RELEASE_FRAC = 0.08 # Audio release percentage (fade out)

# STFT / features
N_FFT = 2048 # The number of FFT's taken (FFT answers how much of a frquency is inside a chunk of audio.) 
HOP = 512 # How far forward you slide when each FFT snapshot is taken (HOP/SR = sec)
MIN_FREQ_HZ = 40  # Ignore frequencies below this

# Model
SOLVER = "lbfgs" # Method used to train logistic regression model
MAX_ITER = 3000 # The amount of times comparisons are made for the whole model to converge on a formula

# Output paths
OUT_ASSETS_DIR = Path("../src/assets") # Path for React app access
OUT_JSON = OUT_ASSETS_DIR / "chord_model.json" # Output of ML model
OUT_JOBLIB = Path("chord_clf.joblib") # Output of ML Model

# =========================

PITCHES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def midi_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

def chord_midis(root_pc: int, is_minor: bool):
    base_midi = random.choice(BASE_MIDI_CHOICES)
    base_pc = base_midi % 12
    delta = (root_pc - base_pc) % 12
    root_midi = base_midi + delta
    third = 3 if is_minor else 4
    return [root_midi, root_midi + third, root_midi + 7]

def synth_chord(root_pc=0, is_minor=False):
    n = int(SR * DUR)
    t = np.arange(n, dtype=np.float32) / SR

    notes = chord_midis(root_pc, is_minor)

    # envelope
    attack = int(ATTACK_FRAC * n)
    release = int(RELEASE_FRAC * n)
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    env[-release:] = np.linspace(1, 0, release, dtype=np.float32)

    y = np.zeros(n, dtype=np.float32)
    harmonics = random.choice(HARMONICS_CHOICES)
    detune_cents = random.uniform(*DETUNE_CENTS_RANGE)

    for midi in notes:
        freq = midi_freq(midi) * (2 ** (detune_cents / 1200.0))
        amp = random.uniform(*AMP_RANGE)
        for k in range(1, harmonics + 1):
            y += (amp / (k ** 1.2)) * np.sin(2 * np.pi * (freq * k) * t).astype(np.float32)

    y += NOISE_STD * np.random.randn(n).astype(np.float32)
    y *= env
    y /= (np.max(np.abs(y)) + 1e-9)
    return y

def stft_mag(y):
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    window = np.hanning(N_FFT).astype(np.float32)

    frames = []
    for start in range(0, len(y) - N_FFT + 1, HOP):
        frame = y[start:start+N_FFT] * window
        spec = np.fft.rfft(frame)
        frames.append(np.abs(spec).astype(np.float32))

    return np.stack(frames, axis=1)  # [freq_bins, time]

def chroma_from_stft(S):
    freqs = np.fft.rfftfreq(N_FFT, d=1.0/SR)
    chroma = np.zeros((12, S.shape[1]), dtype=np.float32)

    for i, f in enumerate(freqs):
        if f < MIN_FREQ_HZ:
            continue
        midi = 69 + 12 * np.log2(f / 440.0)
        pc = int(np.round(midi)) % 12
        chroma[pc, :] += S[i, :]

    denom = chroma.sum(axis=0, keepdims=True) + 1e-9
    chroma /= denom
    return chroma

def features(y):
    S = stft_mag(y)
    C = chroma_from_stft(S)  # [12, T]
    feat = np.concatenate([C.mean(axis=1), C.std(axis=1)], axis=0)  # 24 dims
    return feat.astype(np.float32)

def build_dataset():
    random.seed(SEED)
    np.random.seed(SEED)

    labels = [p for p in PITCHES] + [p + "m" for p in PITCHES]
    X, y = [], []

    # majors
    for root in range(12):
        for _ in range(SAMPLES_PER_CLASS):
            audio = synth_chord(root_pc=root, is_minor=False)
            X.append(features(audio))
            y.append(root)

    # minors
    for root in range(12):
        for _ in range(SAMPLES_PER_CLASS):
            audio = synth_chord(root_pc=root, is_minor=True)
            X.append(features(audio))
            y.append(12 + root)

    return np.stack(X), np.array(y, dtype=np.int64), labels

def main():
    X, y, labels = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = LogisticRegression(
        solver=SOLVER,
        max_iter=MAX_ITER,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=labels))

    joblib.dump(clf, OUT_JOBLIB)

    export = {
        "labels": labels,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "config": {  # optional: saves your settings inside the JSON
            "SAMPLES_PER_CLASS": SAMPLES_PER_CLASS,
            "SR": SR,
            "DUR": DUR,
            "N_FFT": N_FFT,
            "HOP": HOP,
            "NOISE_STD": NOISE_STD,
            "SEED": SEED
        }
    }

    OUT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(export, f)

    print(f"Saved model to {OUT_JSON}")

if __name__ == "__main__":
    main()