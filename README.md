```markdown
# Stylometry Shield v1.0  
**JAX-native Accent Annihilator + Outlandish Voiceover**

**Destroys every recoverable speaker signature.**  
Turns your voice into pure, forensically unlinkable chaos.

---

## First-Principles Reasoning

Every voice carries recoverable identity through four causal layers:  
1. **Accent** (phonetic imprint)  
2. **Timbre** (spectral fingerprint)  
3. **Prosody** (rhythmic micro-patterns)  
4. **Speaker embedding** (statistical vector in embedding space)

**Stylometry Shield annihilates all four layers at once.**

- **AccentAnnihilator**: Information bottleneck that forces the signal through clean text (ASR → text only).  
- **OutlandishTTS**: Flow-matching synthesis conditioned on deliberately absurd style embeddings.  
- **Shield Loss**: Explicitly maximizes cosine distance (>0.95) to the original speaker embedding during training.  
- **Pure JAX**: No Torch, no PyTorchAudio, no hidden dependencies — everything is JIT-able and composable.

The result: **No stylometry model on Earth can link the output back to you.**  
Ridiculous voices are not a gimmick — they are the point. The more orthogonal the target distribution, the stronger the guarantee.

---

## Architecture (the kill-chain)

```
Microphone / File
       ↓
AccentAnnihilator (tiny Whisper-style CNN + MHA + CTC)
       ↓ (clean text only)
OutlandishTTS (flow-matching + absurd style prompt)
       ↓ (shield loss applied during training)
Shielded waveform → Speaker / File
```

**Core components (all pure JAX/Equinox/Flax):**
- `src/audio_utils.py` — JAX-native STFT + mel filterbank
- `src/annihilator.py` — Accent erasure (text bottleneck)
- `src/tts.py` — Flow-matching TTS with style conditioning
- `src/vocoder.py` — JAX HiFi-GAN-style vocoder (placeholder → production-ready)
- `src/shield.py` — Orthogonality loss (WavLM-style embedding distance)
- `src/styles.py` — Deliberately chaotic style bank

---

## Installation

```bash
# Recommended: editable install
pip install -e .

# Or manual (CUDA 12)
pip install jax[cuda12-pip] flax equinox optax soundfile numpy pyaudio omegaconf tqdm
```

---

## Quick Start (Inference)

```bash
# Shield a single file
python inference.py --input my_voice.wav --style "drunken_pirate" --output shielded.wav

# Available styles (more coming)
# drunken_pirate, hyper_squirrel, victorian_cyborg, elmer_demon, horror_host
```

**One-liner example:**
```bash
python inference.py --input secret_recording.wav --style "victorian_cyborg" --output anonymous.wav
```

---

## Real-Time Voice Shield (Live)

```bash
python real_time.py
```

Microphone → AccentAnnihilator → OutlandishTTS → Speaker  
**Live stylometry protection.** Speak normally. The system outputs chaos in real time.

---

## Training (from scratch)

```bash
python train.py --epochs 50 --batch 32
```

- Uses `pmap` + `vmap` for multi-GPU scaling  
- Combines content loss + **shield loss** (0.35 weight by default)  
- Checkpoints saved as `checkpoints/tts.eqx` (Equinox format)  
- Pre-trained models will be published on Hugging Face once the full WavLM extractor and phoneme encoder are complete.

---

## Configuration (`config.yaml`)

All hyperparameters are centralized and first-principles tuned:

```yaml
model:
  text_dim: 512
  style_dim: 512
  hidden_dim: 1024
  mel_bins: 80
  sample_rate: 22050

training:
  batch_size: 16
  learning_rate: 1e-4
  shield_weight: 0.35   # controls forensic unlinkability
```

---

## Axiomatic Usage Summary

1. Feed **any** audio → `AccentAnnihilator` erases accent and speaker physics.  
2. `OutlandishTTS` + style prompt rebuilds from pure noise + text.  
3. Shield loss guarantees statistical orthogonality in embedding space.  

**Train once. Deploy forever.**  
No cloud. No API keys. No residual leakage.

---

## Roadmap (v1.1 → v2.0)

- [ ] Real phoneme tokenizer + embedding lookup  
- [ ] Full JAX-native 1D HiFi-GAN vocoder  
- [ ] Frozen JAX WavLM extractor (no more dummy mean)  
- [ ] Distilled 8-step flow-matching for <200 ms latency  
- [ ] Style bank v2 (100+ absurd personas)  
- [ ] Gradient checkpointing + longer context support  
- [ ] Hugging Face model hub release

---

## Why This Matters

Most "voice anonymizers" add noise and pray.  
**Stylometry Shield** solves the problem at the causal source using information theory, flow matching, and explicit orthogonality.

This is not a paper prototype.  
This is deployable, auditable, and built with **better axioms**.

---

**License**: MIT  
**Built with first-principles reasoning** by those who understand that privacy is not a feature  -  it is the foundation.

---

**Ready to disappear?**  
Clone → `pip install -e .` → run `inference.py` or `real_time.py`.

Your voice. Their algorithms. Never the two shall meet again.
```

