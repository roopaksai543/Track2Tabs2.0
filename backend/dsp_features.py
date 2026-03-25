import warnings

import numpy as np
import librosa


def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def ensure_min_length(y, min_len=2048):
    if len(y) >= min_len:
        return y.astype(np.float32)

    pad_amount = min_len - len(y)
    y = np.pad(y, (0, pad_amount), mode="constant")
    return y.astype(np.float32)


def harmonic_only(y):
    y = ensure_min_length(y, min_len=2048)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_harm, _ = librosa.effects.hpss(y)

    return y_harm.astype(np.float32)


def extract_feature_sequence(y, sr, hop_length=512):
    y = ensure_min_length(y, min_len=2048)
    y = harmonic_only(y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        chroma_cqt = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        chroma_cens = librosa.feature.chroma_cens(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )

        tonnetz = librosa.feature.tonnetz(
            y=y,
            sr=sr,
        )

    min_frames = min(
        chroma_cqt.shape[1],
        chroma_cens.shape[1],
        tonnetz.shape[1],
    )

    if min_frames <= 0:
        return np.zeros((0, 30), dtype=np.float32)

    chroma_cqt = chroma_cqt[:, :min_frames]
    chroma_cens = chroma_cens[:, :min_frames]
    tonnetz = tonnetz[:, :min_frames]

    feats = np.vstack([chroma_cqt, chroma_cens, tonnetz])   # [features, frames]
    feats = feats.T.astype(np.float32)                      # [frames, features]

    return feats


def get_frame_times(num_frames, sr, hop_length=512):
    return librosa.frames_to_time(
        np.arange(num_frames),
        sr=sr,
        hop_length=hop_length,
    )