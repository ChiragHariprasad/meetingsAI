import pickle
import torch
import numpy as np
import sounddevice as sd
from scipy.spatial.distance import cosine
from speechbrain.inference import EncoderClassifier

DB_PATH = "speaker_db.pkl"
SAMPLE_RATE = 16000
WINDOW_SEC = 2.0
THRESHOLD = 0.70   # tune later
DEVICE = "cpu"     # keep CPU for stability

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

with open(DB_PATH, "rb") as f:
    SPEAKERS = pickle.load(f)

def identify(audio):
    audio = torch.tensor(audio).unsqueeze(0)
    with torch.no_grad():
        emb = classifier.encode_batch(audio).squeeze().cpu()

    best_name = "Unknown"
    best_score = 1.0

    for name, ref_emb in SPEAKERS.items():
        score = cosine(emb, ref_emb)
        if score < best_score:
            best_score = score
            best_name = name

    if best_score > THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score

print("🎤 Listening... Speak naturally")

while True:
    audio = sd.rec(
        int(WINDOW_SEC * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = audio.squeeze()
    speaker, score = identify(audio)
    print(f"➡️ {speaker}  (score={score:.3f})")
