import uuid
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from chord_infer import ChordInferenceEngine
from stem_seperate import separate_stems


APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent  # go up from backend → project root

ML_DIR = ROOT_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "chord_sequence_model.pt"
LABELS_PATH = ARTIFACTS_DIR / "chord_labels.json"

TMP_DIR.mkdir(exist_ok=True)

engine = ChordInferenceEngine(str(MODEL_PATH), str(LABELS_PATH))

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
    tempo = np.asarray(tempo).squeeze()

    if tempo.ndim == 0:
        return float(tempo)

    if tempo.size > 0:
        return float(tempo.flat[0])

    return 0.0


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

        timeline = engine.predict_file(melody_path)

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