import torchaudio as ta
import torch
import os
from huggingface_hub import login
token='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'
# hf_token = os.environ.get("HF_TOKEN")
# if not hf_token:
#     raise ValueError("HF_TOKEN environment variable is required")

login(token=token)
from chatterbox.tts_turbo import ChatterboxTurboTTS
# Load the Turbo model
# Load the Turbo model on Apple Silicon (MPS)
model = ChatterboxTurboTTS.from_pretrained(
    device="mps"
)
# Generate with Paralinguistic Tags
text = 'The streets were quieter these days, a far cry from the bustling metropolis Serenity City used to be. That was all before the aliens arrived—before the world had turned upside down. As I approached the park, I glanced up at the colossal alien vessel that had become an unwelcome fixture in our sky.'

wav = model.generate(text)

ta.save("killquest.wav", wav, model.sr)