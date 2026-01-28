# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note:** This is the chatterbox repo with a few of my personal changes. Its primary workload is in the file `src/chatterbox/tts.py`. I removed the watermark as it made the voice quality poor.

## Project Overview

Chatterbox is an open-source text-to-speech (TTS) and voice conversion library by Resemble AI. It provides three main model variants:
- **Chatterbox-Turbo** (350M params): Fast, efficient TTS with paralinguistic tags (`[laugh]`, `[chuckle]`, etc.)
- **Chatterbox** (500M params): English TTS with CFG and exaggeration controls
- **Chatterbox-Multilingual** (500M params): Supports 23+ languages

## Common Commands

```bash
# Install from source (editable mode)
pip install -e .

# Run Gradio demos
python gradio_tts_app.py          # Original Chatterbox TTS
python gradio_tts_turbo_app.py    # Turbo model
python gradio_vc_app.py           # Voice conversion

# Run example scripts
python example_tts.py
python example_tts_turbo.py
python example_vc.py
```

## Architecture

### Core Pipeline

The TTS pipeline consists of two stages:
1. **T3 (Token-To-Token)**: LLM-based model that converts text tokens to speech tokens
2. **S3Gen**: Converts speech tokens to audio waveforms via CFM (Conditional Flow Matching) + HiFiGAN vocoder

### Main Entry Points

- `src/chatterbox/tts.py` - `ChatterboxTTS`: English-only TTS with CFG support
- `src/chatterbox/tts_turbo.py` - `ChatterboxTurboTTS`: Fast 350M model, uses GPT-2 backbone
- `src/chatterbox/mtl_tts.py` - `ChatterboxMultilingualTTS`: 23-language support
- `src/chatterbox/vc.py` - `ChatterboxVC`: Voice conversion (audio-to-audio)

### Model Components (`src/chatterbox/models/`)

- `t3/t3.py` - Token-to-token transformer (LLaMA or GPT-2 backbone)
- `s3gen/s3gen.py` - Speech token to waveform decoder (`S3Token2Mel` → `S3Token2Wav`)
- `s3tokenizer/` - Speech tokenizer (converts 16kHz audio to tokens)
- `voice_encoder/` - Speaker embedding extraction
- `tokenizers/` - Text tokenizers (English: `EnTokenizer`, Multilingual: `MTLTokenizer`)

### Key Constants

- `S3_SR = 16000` - Sample rate for speech tokenizer input
- `S3GEN_SR = 24000` - Sample rate for synthesized audio output

### Model Loading Pattern

All TTS classes follow this pattern:
```python
model = ChatterboxTTS.from_pretrained(device="cuda")  # Downloads from HuggingFace
# or
model = ChatterboxTTS.from_local(ckpt_dir, device)    # Load from local path
```

### Generation API

```python
wav = model.generate(
    text,
    audio_prompt_path="reference.wav",  # Optional: voice cloning reference
    exaggeration=0.5,                   # Emotion intensity (original model only)
    cfg_weight=0.5,                     # Classifier-free guidance (original model only)
    temperature=0.8,
    repetition_penalty=1.2,
)
```

### Perth Watermarking

All generated audio includes imperceptible neural watermarks via the `resemble-perth` library. Watermarks can be detected with `perth.PerthImplicitWatermarker().get_watermark()`.
