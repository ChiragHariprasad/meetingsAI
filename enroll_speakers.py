import os
import pickle
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier

DATA_DIR = r"E:\WorkingProjects\MeetingsAI\Data"
OUT_DB = "speaker_db.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE}
)

SAMPLE_RATE = 16000
speaker_db = {}

def load_wav(path):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav

for speaker in os.listdir(DATA_DIR):
    speaker_path = os.path.join(DATA_DIR, speaker)
    if not os.path.isdir(speaker_path):
        continue

    embeddings = []

    for file in os.listdir(speaker_path):
        if not file.lower().endswith(".wav"):
            continue

        wav = load_wav(os.path.join(speaker_path, file))
        with torch.no_grad():
            emb = classifier.encode_batch(wav.to(DEVICE))
        embeddings.append(emb.squeeze().cpu())

    speaker_db[speaker] = torch.stack(embeddings).mean(dim=0)
    print(f"✅ Enrolled {speaker} ({len(embeddings)} samples)")

with open(OUT_DB, "wb") as f:
    pickle.dump(speaker_db, f)

print("\n🎉 Speaker enrollment complete")
