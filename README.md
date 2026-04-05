<div align="center">

## 🎙️ VibeVoice: A Frontier Long Conversational Text-to-Speech Model
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=microsoft)](https://microsoft.github.io/VibeVoice)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f)
[![Technical Report](https://img.shields.io/badge/Technical-Report-red?logo=adobeacrobatreader)](https://arxiv.org/pdf/2508.19205)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb)
[![Live Playground](https://img.shields.io/badge/Live-Playground-green?logo=gradio)](https://aka.ms/VibeVoice-Demo)

</div>
<!-- <div align="center">
<img src="Figures/log.png" alt="VibeVoice Logo" width="200">
</div> -->

<div align="center">
<img src="Figures/VibeVoice_logo.png" alt="VibeVoice Logo" width="300">
</div>

## 🚀 VibeVoice Dialogue Generation (main.py)

### What's New (2024-12-14)

- **🔄 True VRAM Cleanup in LOD Mode**: Worker-based multiprocessing architecture that actually frees GPU memory
- **⚡ Process Isolation**: Each generation runs in a separate worker process that gets terminated after completion
- **🎯 Proper EOS Detection**: Worker processes use audio streaming to stop generation at natural speech endpoints
- **💾 Memory Efficiency**: VRAM drops to near-zero between generations (not just "unreserved" but fully reclaimed by OS)
- **🏗️ Architectural Improvement**: Solves the long-standing issue of PyTorch's CUDA allocator holding onto reserved memory

### What's New (2025-01-13)

- **🎤 Vocal Isolation**: AI-powered vocal isolation using [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) to remove background music/noise from voice samples
- **Enabled by Default**: The "Isolate input voices" option is on by default for cleaner voice cloning
- **Auto Model Download**: Vocal isolation model automatically downloads from HuggingFace on first use
- **VRAM Efficient**: Isolation model is loaded, used, and unloaded before main generation to maximize available VRAM
- **Debug Mode**: Use `--debug` to save voice samples at each processing stage to `custom_voices/debug/`

### What's New (2025-09-21)

- Dedicated AI Chat interface separate from the main script editor
- Removed the "Regenerate Last" button (simpler, safer flow)
- New "Feeling Lucky" button: one-click AI script + audio generation
- Chat history with restore/delete and session persistence
- Previous-round memory for better LLM script continuity; repeated prompts trigger a remix variation
- More robust handling when the LLM returns fewer speakers than selected (no hard failure)
- General UX/layout refinements for a smoother conversational scripting experience

### 🔧 Audio Quality Improvements (2025-01-XX)

- **Fixed Audio Truncation Issue**: Resolved the problem where VibeVoice would cut off the last couple syllables ~60% of the time
- **Delayed EOS Processing**: Implemented intelligent end-of-sequence handling that allows audio chunks to complete naturally before termination
- **Silence Buffer**: Added 267ms of silence at the end of generated audio to eliminate "cut off" feeling
- **Improved Generation Flow**: Audio now ends at proper chunk boundaries, preserving full words and syllables
- **Audio Gain Control**: Simple gain adjustment directly in the audio player
- **Branch**: Changes implemented on `fix-early-eos` branch for testing and validation

#### Technical Implementation Details

**File Modified**: `vibevoice/modular/modeling_vibevoice_inference.py`

1. **Delayed EOS Processing**:
   - Added `pending_finish_tags` tensor to track samples that hit EOS but need to complete current audio chunk
   - Modified EOS detection to mark samples as "pending finish" instead of immediate termination
   - Added logic to properly terminate audio streams only after chunk completion
   - Handles edge cases where samples hit EOS without generating audio chunks

2. **Silence Buffer**:
   - Added 2 silent chunks (266.67ms total) at the end of each audio output
   - Uses `torch.zeros_like()` to create silent audio with matching chunk dimensions
   - Concatenates silence buffer to final audio before returning results

3. **Audio Gain Control & Trimming**:
   - **Gain Control**: Simple gain slider (-20dB to +20dB) directly in the audio player
   - **Real-time Adjustment**: Gain changes apply immediately with 0.1dB precision
   - **Reset Function**: Use the reset button in the audio player to return to 0dB gain
   - **Audio Trimming**: Built-in trimming support using Gradio's audio player controls
   - **Smart Caching**: Gain always applies to original audio (prevents compounding artifacts)
   - **Smooth Processing**: Soft clipping prevents clicks and pops during gain adjustments

**Files Modified**: 
- `vibevoice/modular/modeling_vibevoice_inference.py` (EOS fix + silence buffer)
- `main.py` (audio gain control)

**Impact**: Reduced audio truncation from ~60% to ~10%, with elimination of "cut off" feeling at audio endings, plus built-in audio editing capabilities.

A comprehensive Gradio interface for generating high-quality multi-speaker dialogue audio using VibeVoice models. This tool provides an intuitive web interface for creating conversational audio content with advanced features and controls.

### ✨ Features

- **Multi-Speaker Support**: Generate dialogue with up to 4 distinct speakers
- **Model Selection**: Choose between VibeVoice-7B-Preview and VibeVoice-1.5B models
- **Vocal Isolation**: AI-powered removal of background music/noise from voice samples (enabled by default)
- **Voice Normalization**: Automatically normalize voice sample volumes for consistent audio quality
- **Advanced Settings**: Fine-tune generation parameters (CFG scale, diffusion steps, temperature, etc.)
- **AI Script Generation**: Generate dialogue scripts using OpenAI GPT-4.1-mini or compatible servers
- **OpenAI-Compatible Servers**: Support for local and third-party OAI-compatible LLM servers
- **Load-on-Demand (LOD)**: Worker-based architecture that truly frees VRAM after each generation
- **Offline Mode**: Run without internet using cached Hugging Face models
- **Streaming Audio**: Real-time audio generation with live streaming support
- **Audio Gain Control**: Simple gain adjustment directly in the audio player
- **Audio Trimming**: Built-in trimming support using the audio player controls
- **Custom Voices**: Support for custom voice samples in organized subdirectories

### 🔄 Load-on-Demand (LOD) Mode Architecture

The `--lod` flag enables a specialized worker-based architecture that provides **true VRAM cleanup** after each generation:

#### How It Works

1. **Lightweight Main Process**: The Gradio UI runs in the main process with minimal memory footprint
2. **Worker Process Spawning**: When generation starts, a separate worker process is spawned
3. **Model Loading**: The worker process loads the model (~18GB VRAM) and performs generation
4. **Process Termination**: After generation completes, the worker process is terminated
5. **OS Memory Reclaim**: The operating system forcibly reclaims ALL GPU memory from the terminated process

#### Key Benefits

- ✅ **True VRAM Cleanup**: Memory drops to near-zero between generations (not just "unreserved")
- ✅ **Faster Startup**: Main UI starts immediately without loading the model
- ✅ **No Memory Leaks**: Fresh process for each generation eliminates accumulation
- ✅ **Proper EOS Detection**: Worker uses audio streamer to stop generation at natural speech endpoints

#### Technical Details

Traditional model unloading only marks GPU memory as "unreserved" but PyTorch's CUDA allocator holds onto it for performance. When you kill a process, the OS kernel forcibly reclaims all GPU allocations—this is the **only reliable way** to truly free PyTorch's reserved CUDA memory.

#### Trade-offs

- **No Real-Time Streaming**: LOD mode returns complete audio after generation (not chunk-by-chunk)
- **Process Spawn Overhead**: ~1-2 seconds startup time per generation
- **Can't Stop Mid-Generation**: Must wait for generation to complete

**Recommended for**: Systems with limited VRAM, running multiple AI services, or when you need guaranteed memory cleanup between generations.

### 🎤 Vocal Isolation

VibeVoice includes AI-powered vocal isolation to automatically remove background music, noise, and other non-vocal audio from your voice samples. This results in cleaner voice cloning and more consistent output.

#### How It Works

The vocal isolation feature uses the [Mel-Band-Roformer](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model) model, a state-of-the-art audio source separation model. The implementation is based on the [ComfyUI-MelBandRoFormer](https://github.com/kijai/ComfyUI-MelBandRoFormer) project.

1. **Auto-Download**: On first use, the model (~100MB) is automatically downloaded from HuggingFace
2. **Processing**: Voice samples are processed to extract clean vocals before generation
3. **VRAM Efficient**: The isolation model is loaded, used, and immediately unloaded before main generation
4. **Order of Operations**: Isolation runs first, then normalization (if enabled)

#### Settings

Located in **🎤 Voice Input Settings** accordion:

| Option | Default | Description |
|--------|---------|-------------|
| **Isolate input voices** | ✅ Enabled | Remove background music/noise using AI vocal isolation |
| **Normalize voices** | ❌ Disabled | Normalize volume levels across all voice samples |

#### Debug Mode

When running with `--debug`, voice samples are saved to `custom_voices/debug/` at each processing stage:

- `{speaker_name}_original.wav` - Raw input before any processing
- `{speaker_name}_isolated.wav` - After vocal isolation (if enabled)
- `{speaker_name}_normalized.wav` - After normalization (if enabled)

This helps verify the vocal isolation quality and troubleshoot any issues.

#### Requirements

The vocal isolation feature requires additional dependencies (installed automatically with `pip install -e .`):

```bash
pip install rotary-embedding-torch einops
```

### 🎯 Usage

```bash
# Basic usage
python main.py

# With load-on-demand mode (faster startup, true VRAM cleanup)
python main.py --lod

# With debug mode
python main.py --debug

# Custom port
python main.py --port 8080

# Use local OpenAI-compatible server
python main.py --lod --debug \
  --script-ai-url "http://localhost:11434/v1" \
  --script_ai_model "qwen2.5:7b-instruct" \
  --script_ai_api_key ""

# Use offline mode for Hugging Face models
python main.py --lod --hf-offline

# Custom cache directory
python main.py --lod --hf-cache-dir "/path/to/cache"
```

### 🔧 Setup

1. **Install dependencies**: Follow the installation instructions below
2. **Configure API keys**: Copy `.env-sample` to `.env` and add your API keys
3. **Add custom voices**: Place voice samples in the `custom_voices/` directory (supports subdirectories)
4. **Run the interface**: 
   - **Windows**: Double-click `run_vibevoice.bat` (easiest)
   - **Other platforms**: Execute `python main.py`

### 🎵 Audio Controls

VibeVoice includes built-in audio editing capabilities directly in the player:

#### Gain Control
- **Gain Slider**: Adjust audio volume from -20dB to +20dB with 0.1dB precision
- **Real-time Updates**: Changes apply immediately without reprocessing
- **Reset Button**: Click the reset button in the audio player to return to 0dB gain
- **Smart Processing**: Gain always applies to the original audio, preventing artifacts

#### Audio Trimming
- **Built-in Trimmer**: Use Gradio's audio player controls to select and trim audio segments
- **Visual Selection**: Click and drag on the waveform to select the desired portion
- **Download Trimmed**: Download only the selected portion of the audio
- **Gain Integration**: Gain adjustments work seamlessly with trimmed audio

### 🤖 AI Script Generation

VibeVoice supports AI-powered script generation using OpenAI or compatible servers. You can configure this via CLI arguments or environment variables.

#### Quick Setup
1. **Copy the sample file**: `cp .env-sample .env`
2. **Edit `.env`**: Add your API keys and preferred settings
3. **Run**: `python main.py`

#### OpenAI Platform (Default)
```bash
# .env file
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-4.1-mini  # Optional: change default model
```

#### OpenAI-Compatible Servers
Support for local and third-party servers (Ollama, LM Studio, vLLM, etc.):

**Via CLI (temporary):**
```bash
# Local Ollama server
python main.py --lod --debug \
  --script-ai-url "http://localhost:11434/v1" \
  --script_ai_model "qwen2.5:7b-instruct" \
  --script_ai-api-key ""

# Remote server with API key
python main.py --lod --debug \
  --script-ai-url "https://api.example.com/v1" \
  --script_ai_model "myorg/model-name" \
  --script_ai-api-key "your-api-key"

# Google Gemini API
python main.py --lod --debug \
  --script-ai-url "https://generativelanguage.googleapis.com/v1beta/openai" \
  --script_ai_model "gemini-2.5-flash" \
  --script_ai-api-key "your-gemini-api-key"
```

**Via .env file (persistent):**
```bash
# .env file
SCRIPT_AI_URL=http://localhost:11434/v1
SCRIPT_AI_MODEL=qwen2.5:7b-instruct
SCRIPT_AI_API_KEY=

# Optional: override default OpenAI model
OPENAI_MODEL=gpt-4.1-mini
```

#### Configuration Precedence
Settings are applied in this order (later overrides earlier):
1. **Defaults**: `gpt-4.1-mini` model, OpenAI platform
2. **Environment variables**: `.env` file settings
3. **CLI arguments**: Command-line flags (highest priority)

#### Supported Server Features
- **Chat Completions**: Full support for `/v1/chat/completions` endpoint
- **Multiple Response Formats**: Handles `choices[].message.content`, `choices[].text`, and `choices[].content`
- **Auto URL Normalization**: Automatically appends `/v1` if missing
- **Flexible API Keys**: Empty keys supported for local servers

### 🔄 Offline Mode

Run VibeVoice without internet access using cached models:

```bash
# Force offline mode
python main.py --lod --hf-offline

# Use custom cache directory
python main.py --lod --hf-offline --hf-cache-dir "/shared/cache"

# Environment variable (alternative)
export HF_HUB_OFFLINE=1
python main.py --lod
```

### 📁 Voice Organization

- **Demo voices**: Located in `demo/voices/` (included with the project)
- **Custom voices**: Place in `custom_voices/` directory
- **Subdirectories**: Organize voices into subdirectories (e.g., `custom_voices/characters/`, `custom_voices/narrators/`)
- **Supported formats**: WAV, MP3, FLAC, OGG, M4A, AAC

### 🔑 API Key Requirements

**OpenAI Platform**: Requires `OPENAI_API_KEY` in `.env` file
**Custom Servers**: API key optional (many local servers don't require one)

Example `.env` file:
```bash
# OpenAI platform (required for default)
OPENAI_API_KEY=sk-your-openai-key-here

# Custom server (optional)
SCRIPT_AI_URL=http://localhost:11434/v1
SCRIPT_AI_MODEL=qwen2.5:7b-instruct
SCRIPT_AI_API_KEY=

# Google Gemini API (alternative)
# SCRIPT_AI_URL=https://generativelanguage.googleapis.com/v1beta/openai
# SCRIPT_AI_MODEL=gemini-2.5-flash
# SCRIPT_AI_API_KEY=your-gemini-api-key

# Default model override (optional)
OPENAI_MODEL=gpt-4.1-mini
```

---

VibeVoice is a novel framework designed for generating **expressive**, **long-form**, **multi-speaker** conversational audio, such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

The model can synthesize speech up to **90 minutes** long with up to **4 distinct speakers**, surpassing the typical 1-2 speaker limits of many prior models. 


<p align="left">
  <img src="Figures/MOS-preference.png" alt="MOS Preference Results" height="260px">
  <img src="Figures/VibeVoice.jpg" alt="VibeVoice Overview" height="250px" style="margin-right: 10px;">
</p>

### 🔥 News

- **[2025-08-26] 🎉 We Opensource the [VibeVoice-7B-Preview](https://huggingface.co/vibevoice/VibeVoice-7B) model weights!**

### 📋 TODO

- [ ] Merge models into official Hugging Face repository
- [ ] Release example training code and documentation

### 🎵 Demo Examples


**Video Demo**

We produced this video with [Wan2.2](https://github.com/Wan-Video/Wan2.2). We sincerely appreciate the Wan-Video team for their great work.

**English**
<div align="center">

https://github.com/user-attachments/assets/0967027c-141e-4909-bec8-091558b1b784

</div>


**Chinese**
<div align="center">

https://github.com/user-attachments/assets/322280b7-3093-4c67-86e3-10be4746c88f

</div>

**Cross-Lingual**
<div align="center">

https://github.com/user-attachments/assets/838d8ad9-a201-4dde-bb45-8cd3f59ce722

</div>

**Spontaneous Singing**
<div align="center">

https://github.com/user-attachments/assets/6f27a8a5-0c60-4f57-87f3-7dea2e11c730

</div>


**Long Conversation with 4 people**
<div align="center">

https://github.com/user-attachments/assets/a357c4b6-9768-495c-a576-1618f6275727

</div>

For more examples, see the [Project Page](https://microsoft.github.io/VibeVoice).

Try your own samples at [Colab](https://colab.research.google.com/github/microsoft/VibeVoice/blob/main/demo/VibeVoice_colab.ipynb) or [Demo](https://aka.ms/VibeVoice-Demo).



## Models
| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-0.5B-Streaming | - | - | On the way |
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-7B-Preview| 32K | ~45 min | [HF link](https://huggingface.co/vibevoice/VibeVoice-7B) |

## Installation

Pick **one** of the paths below (Direct Install or Docker) and follow it start to finish. Most people should use the **Direct Install**.

### Prerequisites

Before you begin, make sure you have the following installed on your system:

| Requirement | Why you need it | How to check |
|---|---|---|
| **Python 3.8+** | Runs the application | `python --version` (or `python3 --version`) |
| **pip** | Installs Python packages | `pip --version` (or `pip3 --version`) |
| **Git** | Clones the repository | `git --version` |
| **ffmpeg** | Processes audio files | `ffmpeg -version` |
| **NVIDIA GPU + drivers** (recommended) | Fast speech generation | `nvidia-smi` |

<details>
<summary><b>How to install prerequisites if you're missing any</b></summary>

**Python & pip** — Download from [python.org](https://www.python.org/downloads/). On Linux you can use your package manager:
```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv -y   # Debian/Ubuntu
```

**Git** — Download from [git-scm.com](https://git-scm.com/downloads), or on Linux:
```bash
sudo apt update && sudo apt install git -y   # Debian/Ubuntu
```

**ffmpeg** — Download from [ffmpeg.org](https://ffmpeg.org/download.html), or on Linux:
```bash
sudo apt update && sudo apt install ffmpeg -y   # Debian/Ubuntu
```

**NVIDIA drivers** — Follow the [NVIDIA driver install guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/) for your OS. VibeVoice can fall back to CPU or Apple Silicon (MPS) if no NVIDIA GPU is available, but generation will be significantly slower.
</details>

---

### 💻 Direct Installation (Recommended for most users)

Open a terminal (Command Prompt or PowerShell on Windows) and run these commands **one group at a time**.

#### Step 1 — Clone the repository

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
```

This downloads the source code and moves you into the project folder.

#### Step 2 — Create and activate a virtual environment

A virtual environment keeps VibeVoice's dependencies separate from the rest of your system so nothing conflicts.

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> After activation your terminal prompt should show `(venv)` at the beginning. All remaining commands assume the virtual environment is active.

#### Step 3 — Install PyTorch

PyTorch is the machine-learning framework VibeVoice is built on. Install the version that matches your hardware:

**NVIDIA GPU (CUDA 12.1) — recommended:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only or Apple Silicon (MPS):**
```bash
pip install torch torchvision torchaudio
```

> Not sure which to pick? Run `nvidia-smi`. If it prints GPU info, use the CUDA version. If it errors, use the CPU version. For other CUDA versions, see the [PyTorch install matrix](https://pytorch.org/get-started/locally/).

#### Step 4 — Install VibeVoice and all remaining dependencies

```bash
pip install -e .
```

This reads the project's `pyproject.toml` file and installs everything VibeVoice needs (transformers, gradio, librosa, openai, etc.). The `-e` ("editable") flag means Python uses the code right here in this folder, so any updates you `git pull` are picked up automatically.

#### Step 5 (optional) — Install FlashAttention2 for faster generation

FlashAttention2 speeds up the model on NVIDIA GPUs. It is **not required** — VibeVoice falls back to a compatible attention implementation automatically.

```bash
pip install flash-attn --no-build-isolation
```

> **Windows users:** Building from source can be tricky. Pre-built wheels are available at [sunsetcoder/flash-attention-windows](https://github.com/sunsetcoder/flash-attention-windows).

#### Step 6 — Configure your API key (for AI script generation)

The AI scriptwriter needs an LLM API key. Copy the sample config and edit it:

**Linux / macOS:**
```bash
cp .env-sample .env
nano .env          # or open .env in any text editor
```

**Windows:**
```cmd
copy .env-sample .env
notepad .env
```

Inside `.env`, replace `your-open-ai-key` with your actual API key. If you don't have an OpenAI key, you can use a **free Google Gemini key** — see the comments in `.env-sample` for details. AI script generation is optional; VibeVoice works without it.

#### Step 7 — Run VibeVoice

```bash
python main.py
```

After a moment you'll see a local URL (usually `http://localhost:7860`). Open it in your browser and you're ready to go.

> **Tip:** Add `--lod` for load-on-demand mode, which uses much less VRAM when idle:
> ```bash
> python main.py --lod
> ```

#### Verify everything is working

If something went wrong during install, you can run a quick sanity check:

```bash
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import vibevoice; print('VibeVoice package OK')"
```

---

### 🐳 Docker Installation (Alternative)

Docker is useful if you want a pre-configured CUDA environment without installing drivers on the host.

```bash
# 1. Launch the NVIDIA PyTorch container (24.07 / 24.10 / 24.12 verified)
sudo docker run --privileged --net=host --ipc=host \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --gpus all --rm -it nvcr.io/nvidia/pytorch:24.07-py3

# 2. Inside the container, clone and install
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .

# 3. (Optional) Install FlashAttention2 if not bundled in your container
pip install flash-attn --no-build-isolation

# 4. Configure your API key and run
cp .env-sample .env
# edit .env with your API key
python main.py
```

---

### 🪟 Windows Quick Start

For Windows users, we provide a convenient batch script:

1. Follow the **Direct Installation** steps above
2. Ensure `.env` is configured with your API key
3. **Double-click** `run_vibevoice.bat` to launch

The batch script will:
- Check for and activate the virtual environment
- Launch VibeVoice on `http://localhost:7590`
- Show helpful error messages if setup is incomplete

> **Load-on-Demand Mode**: Edit `run_vibevoice.bat`, comment out `python main.py`, and uncomment `python main.py --lod`.

---

### 🔧 Device Compatibility & Fallback Support

VibeVoice supports multiple hardware configurations with automatic fallback:

- **CUDA (NVIDIA GPUs)** — Full support with optional FlashAttention2 for best performance
- **Apple Silicon (MPS)** — Native support for M1/M2/M3/M4 Macs via Metal Performance Shaders
- **CPU** — Works everywhere, but generation is much slower
- **Robust Fallback** — If FlashAttention2 is unavailable, VibeVoice automatically uses PyTorch's built-in SDPA

## Usages

### 🚨 Tips
We observed users may encounter occasional instability when synthesizing Chinese speech. We recommend:

- Using English punctuation even for Chinese text, preferably only commas and periods.
- Using the 7B model variant, which is considerably more stable.

### Usage 1: Launch Gradio demo
```bash
# Make sure ffmpeg is installed (see Prerequisites above)

# For 1.5B model
python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B

# For 7B model
python demo/gradio_demo.py --model_path WestZhang/VibeVoice-Large-pt
```

### Usage 2: Inference from files directly
```bash
# We provide some LLM generated example scripts under demo/text_examples/ for demo
# 1 speaker
python demo/inference_from_file.py --model_path WestZhang/VibeVoice-Large-pt --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

# or more speakers
python demo/inference_from_file.py --model_path WestZhang/VibeVoice-Large-pt --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank
```

## FAQ
#### Q1: Is this a pretrained model?
**A:** Yes, it's a pretrained model without any post-training or benchmark-specific optimizations. In a way, this makes VibeVoice very versatile and fun to use.

#### Q2: Randomly trigger Sounds / Music / BGM.
**A:** As you can see from our demo page, the background music or sounds are spontaneous. This means we can't directly control whether they are generated or not. The model is content-aware, and these sounds are triggered based on the input text and the chosen voice prompt.

Here are a few things we've noticed:
*   If the voice prompt you use contains background music, the generated speech is more likely to have it as well. (The 7B model is quite stable and effective at this—give it a try on the demo!)
*   If the voice prompt is clean (no BGM), but the input text includes introductory words or phrases like "Welcome to," "Hello," or "However," background music might still appear.
*   Spekaer voice related, using "Alice" results in random BGM than others.
*   In other scenarios, the 7B model is more stable and has a lower probability of generating unexpected background music.

In fact, we intentionally decided not to denoise our training data because we think it's an interesting feature for BGM to show up at just the right moment. You can think of it as a little easter egg we left for you.

#### Q3: Text normalization?
**A:** We don't perform any text normalization during training or inference. Our philosophy is that a large language model should be able to handle complex user inputs on its own. However, due to the nature of the training data, you might still run into some corner cases.

#### Q4: Singing Capability.
**A:** Our training data **doesn't contain any music data**. The ability to sing is an emergent capability of the model (which is why it might sound off-key, even on a famous song like 'See You Again'). (The 7B model is more likely to exhibit this than the 1.5B).

#### Q5: Some Chinese pronunciation errors.
**A:** The volume of Chinese data in our training set is significantly smaller than the English data. Additionally, certain special characters (e.g., Chinese quotation marks) may occasionally cause pronunciation issues.

#### Q6: Why use --lod mode? What about VRAM cleanup?
**A:** The `--lod` (Load-on-Demand) mode uses a worker-based multiprocessing architecture that provides **true VRAM cleanup**. Traditional model unloading only marks GPU memory as "unreserved," but PyTorch's CUDA allocator holds onto it for performance. When you use `--lod` mode, each generation runs in a separate worker process that gets terminated after completion, forcing the OS to reclaim ALL GPU memory. This is the only reliable way to truly free reserved CUDA memory. 

**Use --lod mode when:**
- You have limited VRAM and need guaranteed memory cleanup between generations
- You're running multiple AI services on the same GPU
- You want faster startup (UI loads without waiting for model)
- You don't need real-time audio streaming (LOD returns complete audio after generation)

## Risks and limitations

Potential for Deepfakes and Disinformation: High-quality synthetic speech can be misused to create convincing fake audio content for impersonation, fraud, or spreading disinformation. Users must ensure transcripts are reliable, check content accuracy, and avoid using generated content in misleading ways. Users are expected to use the generated content and to deploy the models in a lawful manner, in full compliance with all applicable laws and regulations in the relevant jurisdictions. It is best practice to disclose the use of AI when sharing AI-generated content.

English and Chinese only: Transcripts in languages other than English or Chinese may result in unexpected audio outputs.

Non-Speech Audio: The model focuses solely on speech synthesis and does not handle background noise, music, or other sound effects.

Overlapping Speech: The current model does not explicitly model or generate overlapping speech segments in conversations.

We do not recommend using VibeVoice in commercial or real-world applications without further testing and development. This model is intended for research and development purposes only. Please use responsibly.

## Acknowledgments

We would like to thank the following contributors for their valuable work that enhanced VibeVoice's compatibility and performance:

### Device Compatibility & Fallback Features
- **Device Detection & Fallback Logic**: Inspired by implementations from the community, particularly [mypapit/VibeVoice](https://github.com/mypapit/VibeVoice) for demonstrating robust device detection and attention mechanism fallbacks.

### FlashAttention2 Windows Support
- [sunsetcoder/flash-attention-windows](https://github.com/sunsetcoder/flash-attention-windows): Pre-built FlashAttention2 wheels for Windows (Python 3.10, CUDA 11.7+)
- [huihui-support/flash-attention-windows](https://github.com/huihui-support/flash-attention-windows): FlashAttention2 wheels for Python 3.10, 3.11, and 3.12
- [ussoewwin/Flash-Attention-2_for_Windows](https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows): FlashAttention2 wheels for Python 3.11 and 3.12
- [felisevan/flash-attention-build](https://github.com/felisevan/flash-attention-build): Additional Windows build support
- [sdbds/flash-attention-for-windows](https://github.com/sdbds/flash-attention-for-windows): Windows compatibility solutions
- [BlackTea-c/flash-attention-windows](https://github.com/BlackTea-c/flash-attention-windows): Community Windows support
- [Creepybits: Flash Attention for ComfyUI on Windows](https://www.zanno.se/flash-attention-for-comfyui/): Windows installation guidance

### Core Technologies
- [PyTorch](https://pytorch.org/): For the implementation of Scaled Dot Product Attention (SDPA) and device management
- [Hugging Face Transformers](https://huggingface.co/transformers/): For the model architecture and attention implementations
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention): For the FlashAttention2 implementation

These contributions have made VibeVoice more accessible across different hardware configurations and operating systems, ensuring a smoother experience for all users.
