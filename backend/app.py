import shutil
import uuid
from pathlib import Path

import soundfile as sf
import librosa


import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from dsp_features import windowed_feature_seq
from chord_infer import ChordModel
from stem_seperate import separate_stems


APP_DIR = Path(__file__).parent
TMP_DIR = APP_DIR / "tmp"
MODELS_DIR = APP_DIR / "models"
MODEL_PATH = MODELS_DIR / "chord_model_2.json" # Model used

TMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

model = ChordModel(str(MODEL_PATH))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_wav_mono(path: str):
    y, sr = sf.read(path, always_2d=False)

    if y.ndim == 2:
        y = y.mean(axis=1)

    return y.astype("float32"), sr


def estimate_tempo(audio, sr):
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

    # librosa may return a scalar or a numpy array depending on version/setup
    if hasattr(tempo, "__len__"):
        tempo = tempo[0] if len(tempo) > 0 else 0.0

    return float(tempo)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    job_dir = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "input_audio.wav"
    input_path = job_dir / filename

    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        drums_path, melody_path = separate_stems(str(input_path), str(job_dir))

        if not Path(drums_path).exists():
            return {"error": "Drums stem missing"}

        if not Path(melody_path).exists():
            return {"error": "Melody stem missing"}

        print("DRUMS STEM:", drums_path)
        print("MELODY STEM:", melody_path)

        drums, sr_d = load_wav_mono(drums_path)
        melody, sr = load_wav_mono(melody_path)

        bpm = estimate_tempo(drums, sr_d)

        seq = windowed_feature_seq(melody, sr, window_sec=1.0, hop_sec=0.5)

        timeline = []
        last = None

        for t0, feat in seq:
            chord, conf, _ = model.predict(feat)

            if conf < 0.40:
                chord = last

            if chord is None:
                continue

            if chord != last:
                timeline.append(
                    {
                        "time": float(t0),
                        "chord": chord,
                        "confidence": float(conf),
                    }
                )

            last = chord

        return {
            "timeline": timeline,
            "tempo": bpm,
            "sampleRate": sr,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # keep temp files for debugging for now
        # uncomment this later when everything works
        # shutil.rmtree(job_dir, ignore_errors=True)
        pass