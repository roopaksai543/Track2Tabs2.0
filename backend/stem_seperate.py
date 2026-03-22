import subprocess
import sys
from pathlib import Path


def separate_stems(audio_path, output_root):
    audio_path = Path(audio_path)
    output_root = Path(output_root)

    stems_out = output_root / "stems"
    stems_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "-o",
        str(stems_out),
        str(audio_path),
    ]

    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:] or p.stdout[-2000:] or "Demucs failed")

    track_name = audio_path.stem
    track_dir = stems_out / "htdemucs" / track_name

    if not track_dir.exists():
        raise RuntimeError(f"Demucs output folder not found: {track_dir}")

    drums = track_dir / "drums.wav"
    other = track_dir / "other.wav"
    vocals = track_dir / "vocals.wav"
    bass = track_dir / "bass.wav"

    if not drums.exists():
        raise RuntimeError(f"Drums stem not found: {drums}")

    if not other.exists():
        raise RuntimeError(f"Other stem not found: {other}")

    # optional debug prints
    print("STEMS FOLDER:", track_dir)
    print("DRUMS:", drums)
    print("OTHER:", other)
    print("VOCALS:", vocals)
    print("BASS:", bass)

    return str(drums), str(other)