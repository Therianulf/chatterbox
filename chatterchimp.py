import torchaudio as ta
import torch
import os
from huggingface_hub import login

#this was just my first attempt at getting the tool to work. commiting it anyways cause fuck it

# hf_token = os.environ.get("HF_TOKEN")
# if not hf_token:
#     raise ValueError("HF_TOKEN environment variable is required")

# login(token=hf_token)
from chatterbox.tts_turbo import ChatterboxTurboTTS
# Load the Turbo model
# Load the Turbo model on Apple Silicon (MPS)
model = ChatterboxTurboTTS.from_pretrained(
    device="mps"
)
# Generate with Paralinguistic Tags
text = 'It was a hell of a sight. The sheer size of the ship boggled the mind—easily spanning thirty city blocks. Its shadow plunged the park and surrounding streets into a perpetual darkness.'

wav = model.generate(text)

ta.save("killquest3.wav", wav, model.sr)