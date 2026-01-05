
import torch, torchaudio, speechbrain, numpy
import huggingface_hub

print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("numpy:", numpy.__version__)
print("hf hub:", huggingface_hub.__version__)
print("audio backends:", torchaudio.list_audio_backends())
print("cuda:", torch.cuda.is_available())

