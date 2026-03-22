import json
import random
from pathlib import Path

import joblib
import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================
# CONFIG (EDIT THESE ONLY)
# =========================
SEED = 10

# Dataset size
SAMPLES_PER_CLASS = 1000
TEST_SIZE = 0.15
RANDOM_STATE = 10

# Audio synthesis
SR = 22050
DUR = 1.0

# More guitar-like octave region
BASE_MIDI_CHOICES = [40, 43, 45, 48, 52, 55, 57, 60]

# Timbre / realism
HARMONICS_CHOICES = [4, 6, 8]
DETUNE_CENTS_RANGE = (-14, 14)
AMP_RANGE = (0.45, 1.0)
NOISE_STD = 0.008

# Envelope
ATTACK_FRAC = 0.01
RELEASE_FRAC = 0.10

# Strum realism
MAX_NOTE_ONSET_JITTER_SEC = 0.035
PICK_NOISE_LEVEL = 0.020

# Reverb realism
REVERB_PROB = 0.75
REVERB_DECAY_RANGE = (0.08, 0.22)
REVERB_MIX_RANGE = (0.05, 0.18)

# Background interference
BACKGROUND_TONE_PROB = 0.35
BACKGROUND_NOISE_STD = 0.004

# Feature settings
N_FFT = 2048
HOP = 512

# Model
SOLVER = "lbfgs"
MAX_ITER = 4000
C_VALUE = 2.0

# Output paths
OUT_ASSETS_DIR = Path("models")
OUT_JSON = OUT_ASSETS_DIR / "chord_model_2.json"
OUT_JOBLIB = Path("chord_clf.joblib")

# =========================

PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def soft_clip(y: np.ndarray, drive: float = 1.6) -> np.ndarray:
    return np.tanh(drive * y) / np.tanh(drive)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


def major_minor_label(root_pc: int, is_minor: bool) -> int:
    return 12 + root_pc if is_minor else root_pc


def chord_label_names():
    return [p for p in PITCHES] + [p + "m" for p in PITCHES]


def build_chord_voicing(root_pc: int, is_minor: bool):
    """
    More realistic voicings:
    - random inversion
    - octave doubling
    - occasionally omit the 5th
    """
    base_midi = random.choice(BASE_MIDI_CHOICES)
    base_pc = base_midi % 12
    delta = (root_pc - base_pc) % 12
    root_midi = base_midi + delta

    third = 3 if is_minor else 4
    triad = [root_midi, root_midi + third, root_midi + 7]

    inversion = random.choice([0, 1, 2])
    voiced = triad[:]

    for _ in range(inversion):
        voiced[0] += 12
        voiced = voiced[1:] + voiced[:1]

    if random.random() < 0.70:
        voiced.append(random.choice(voiced) + 12)

    if random.random() < 0.35:
        voiced.append(root_midi - 12)

    if random.random() < 0.20:
        voiced.append(root_midi + 12)

    if random.random() < 0.12 and len(voiced) >= 3:
        # occasionally thin out the chord a bit
        drop_index = random.randrange(len(voiced))
        voiced.pop(drop_index)

    voiced = sorted(voiced)
    return voiced


def make_note_envelope(n: int) -> np.ndarray:
    attack = max(1, int(ATTACK_FRAC * n))
    release = max(1, int(RELEASE_FRAC * n))

    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    env[-release:] = np.linspace(1, 0, release, dtype=np.float32)

    # extra decay so notes feel less organ-like
    decay_curve = np.linspace(1.0, random.uniform(0.45, 0.80), n, dtype=np.float32)
    env *= decay_curve
    return env


def add_pick_noise(seg: np.ndarray, onset_idx: int):
    pick_len = random.randint(60, 220)
    if onset_idx >= len(seg):
        return

    end = min(len(seg), onset_idx + pick_len)
    burst = np.random.randn(end - onset_idx).astype(np.float32)

    fade = np.linspace(1.0, 0.0, end - onset_idx, dtype=np.float32)
    burst *= fade
    burst *= random.uniform(0.3, 1.0) * PICK_NOISE_LEVEL

    seg[onset_idx:end] += burst


def synth_single_note(midi: int, n_total: int, onset_sec: float) -> np.ndarray:
    y = np.zeros(n_total, dtype=np.float32)

    onset = int(onset_sec * SR)
    if onset >= n_total:
        return y

    n = n_total - onset
    t = np.arange(n, dtype=np.float32) / SR

    env = make_note_envelope(n)

    local_detune_cents = random.uniform(*DETUNE_CENTS_RANGE)
    freq = midi_freq(midi) * (2 ** (local_detune_cents / 1200.0))

    harmonics = random.choice(HARMONICS_CHOICES)
    amp = random.uniform(*AMP_RANGE)

    note = np.zeros(n, dtype=np.float32)

    for k in range(1, harmonics + 1):
        phase = random.uniform(0, 2 * np.pi)
        harmonic_amp = amp / (k ** random.uniform(1.05, 1.45))
        note += harmonic_amp * np.sin(2 * np.pi * (freq * k) * t + phase).astype(np.float32)

    # slight shimmer / beating
    if random.random() < 0.45:
        lfo_rate = random.uniform(3.0, 8.0)
        lfo_depth = random.uniform(0.003, 0.012)
        note *= (1.0 + lfo_depth * np.sin(2 * np.pi * lfo_rate * t)).astype(np.float32)

    note *= env
    y[onset:] += note

    add_pick_noise(y, onset)
    return y


def apply_simple_reverb(y: np.ndarray) -> np.ndarray:
    if random.random() > REVERB_PROB:
        return y

    ir_len = random.randint(int(0.03 * SR), int(0.09 * SR))
    decay = random.uniform(*REVERB_DECAY_RANGE)
    mix = random.uniform(*REVERB_MIX_RANGE)

    t = np.linspace(0.0, decay, ir_len, dtype=np.float32)
    ir = np.exp(-random.uniform(18, 36) * t).astype(np.float32)
    ir *= np.random.uniform(0.85, 1.15, size=ir_len).astype(np.float32)

    # early reflections
    for _ in range(random.randint(2, 5)):
        idx = random.randint(1, ir_len - 1)
        ir[idx] += random.uniform(0.15, 0.5)

    ir /= (np.max(np.abs(ir)) + 1e-9)

    wet = np.convolve(y, ir, mode="full")[: len(y)].astype(np.float32)
    return ((1.0 - mix) * y + mix * wet).astype(np.float32)


def apply_tone_shaping(y: np.ndarray) -> np.ndarray:
    """
    Mild guitar/song-like spectral variation.
    """
    # random pre-emphasis / de-emphasis
    coeff = random.uniform(-0.35, 0.35)
    y2 = np.copy(y)
    y2[1:] = y2[1:] - coeff * y2[:-1]

    # a little low-frequency rumble reduction
    y2 = librosa.effects.preemphasis(y2, coef=random.uniform(0.90, 0.98)).astype(np.float32)

    # blend original back in so it isn't too extreme
    blend = random.uniform(0.15, 0.35)
    return ((1.0 - blend) * y + blend * y2).astype(np.float32)


def add_background_interference(y: np.ndarray) -> np.ndarray:
    n = len(y)

    if random.random() < BACKGROUND_TONE_PROB:
        t = np.arange(n, dtype=np.float32) / SR
        bg_freq = random.choice([110, 147, 196, 220, 247, 294, 330])
        bg = 0.0
        for k in range(1, random.randint(2, 4)):
            bg += (1.0 / k) * np.sin(2 * np.pi * bg_freq * k * t + random.uniform(0, 2 * np.pi))
        y = y + random.uniform(0.002, 0.012) * bg.astype(np.float32)

    y = y + BACKGROUND_NOISE_STD * np.random.randn(n).astype(np.float32)
    return y.astype(np.float32)


def synth_chord(root_pc=0, is_minor=False):
    n_total = int(SR * DUR)
    y = np.zeros(n_total, dtype=np.float32)

    notes = build_chord_voicing(root_pc, is_minor)

    strum_direction = random.choice(["down", "up", "block"])
    onset_times = []

    if strum_direction == "block":
        onset_times = [random.uniform(0.0, 0.010) for _ in notes]
    else:
        base_offsets = sorted(random.uniform(0.0, MAX_NOTE_ONSET_JITTER_SEC) for _ in notes)
        if strum_direction == "up":
            base_offsets = base_offsets[::-1]
        onset_times = base_offsets

    for midi, onset in zip(notes, onset_times):
        y += synth_single_note(midi, n_total, onset)

    y = add_background_interference(y)
    y = apply_simple_reverb(y)
    y = apply_tone_shaping(y)

    # mild saturation / compression
    y = soft_clip(y, drive=random.uniform(1.2, 2.1)).astype(np.float32)

    # final noise
    y += NOISE_STD * np.random.randn(n_total).astype(np.float32)

    return normalize_audio(y)


def safe_chroma_cqt(y):
    try:
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=SR,
            bins_per_octave=36,
            hop_length=HOP
        )
        if chroma.size == 0 or chroma.shape[1] == 0:
            return np.zeros(12, dtype=np.float32)
        return chroma.mean(axis=1).astype(np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def safe_chroma_stft(y):
    try:
        n_fft = min(N_FFT, len(y))
        if n_fft < 32:
            return np.zeros(12, dtype=np.float32)

        n_fft = max(32, 2 ** int(np.floor(np.log2(n_fft))))
        hop_length = min(HOP, max(16, n_fft // 4))

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=SR,
            n_fft=n_fft,
            hop_length=hop_length
        )
        if chroma.size == 0 or chroma.shape[1] == 0:
            return np.zeros(12, dtype=np.float32)
        return chroma.mean(axis=1).astype(np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def features(y):
    feat_cqt = safe_chroma_cqt(y)
    feat_stft = safe_chroma_stft(y)
    feat = np.concatenate([feat_cqt, feat_stft], axis=0)
    return feat.astype(np.float32)


def build_dataset():
    random.seed(SEED)
    np.random.seed(SEED)

    labels = chord_label_names()
    X, y = [], []

    print("Building major chord samples...")
    for root in range(12):
        for _ in range(SAMPLES_PER_CLASS):
            audio = synth_chord(root_pc=root, is_minor=False)
            X.append(features(audio))
            y.append(major_minor_label(root, False))

    print("Building minor chord samples...")
    for root in range(12):
        for _ in range(SAMPLES_PER_CLASS):
            audio = synth_chord(root_pc=root, is_minor=True)
            X.append(features(audio))
            y.append(major_minor_label(root, True))

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, labels


def main():
    print("RUNNING REALISM-BOOSTED TRAINER")
    print(f"SAMPLES_PER_CLASS = {SAMPLES_PER_CLASS}")

    X, y, labels = build_dataset()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = LogisticRegression(
        solver=SOLVER,
        max_iter=MAX_ITER,
        multi_class="multinomial",
        C=C_VALUE,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, target_names=labels))

    joblib.dump(clf, OUT_JOBLIB)

    export = {
        "labels": labels,
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "config": {
            "SAMPLES_PER_CLASS": SAMPLES_PER_CLASS,
            "SR": SR,
            "DUR": DUR,
            "N_FFT": N_FFT,
            "HOP": HOP,
            "NOISE_STD": NOISE_STD,
            "SEED": SEED,
            "realism_augmented": True
        }
    }

    OUT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(export, f)

    print(f"Saved model to {OUT_JSON}")
    print(f"Saved joblib to {OUT_JOBLIB}")


if __name__ == "__main__":
    main()