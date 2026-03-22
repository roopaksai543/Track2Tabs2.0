import numpy as np
import librosa


def _safe_chroma_cqt(seg, sr):
    try:
        chroma = librosa.feature.chroma_cqt(
            y=seg,
            sr=sr,
            bins_per_octave=36
        )
        if chroma.shape[1] == 0:
            return np.zeros(12, dtype=np.float32)
        return np.mean(chroma, axis=1).astype(np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def _safe_chroma_stft(seg, sr):
    try:
        n_fft = min(2048, len(seg))
        if n_fft < 32:
            return np.zeros(12, dtype=np.float32)

        n_fft = max(32, 2 ** int(np.floor(np.log2(n_fft))))
        hop_length = max(16, n_fft // 4)

        chroma = librosa.feature.chroma_stft(
            y=seg,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length
        )
        if chroma.shape[1] == 0:
            return np.zeros(12, dtype=np.float32)
        return np.mean(chroma, axis=1).astype(np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def windowed_feature_seq(y, sr, window_sec=1.0, hop_sec=0.5):
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)

    if win <= 0 or hop <= 0:
        return []

    seq = []

    for start in range(0, max(1, len(y) - win + 1), hop):
        seg = y[start:start + win]

        if len(seg) < max(32, win // 4):
            continue

        feat_cqt = _safe_chroma_cqt(seg, sr)
        feat_stft = _safe_chroma_stft(seg, sr)

        feat = np.concatenate([feat_cqt, feat_stft]).astype(np.float32)

        seq.append((start / sr, feat))

    return seq