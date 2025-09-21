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

### What's New (2025-09-21)

- Dedicated AI Chat interface separate from the main script editor
- Removed the "Regenerate Last" button (simpler, safer flow)
- New "Feeling Lucky" button: one-click AI script + audio generation
- Chat history with restore/delete and session persistence
- Previous-round memory for better LLM script continuity; repeated prompts trigger a remix variation
- More robust handling when the LLM returns fewer speakers than selected (no hard failure)
- General UX/layout refinements for a smoother conversational scripting experience

A comprehensive Gradio interface for generating high-quality multi-speaker dialogue audio using VibeVoice models. This tool provides an intuitive web interface for creating conversational audio content with advanced features and controls.

### ✨ Features

- **Multi-Speaker Support**: Generate dialogue with up to 4 distinct speakers
- **Model Selection**: Choose between VibeVoice-7B-Preview and VibeVoice-1.5B models
- **Voice Normalization**: Automatically normalize voice sample volumes for consistent audio quality
- **Advanced Settings**: Fine-tune generation parameters (CFG scale, diffusion steps, temperature, etc.)
- **AI Script Generation**: Generate dialogue scripts using OpenAI GPT-4.1-mini or compatible servers
- **OpenAI-Compatible Servers**: Support for local and third-party OAI-compatible LLM servers
- **Load-on-Demand**: Option to load models only when needed to save VRAM
- **Offline Mode**: Run without internet using cached Hugging Face models
- **Streaming Audio**: Real-time audio generation with live streaming support
- **Custom Voices**: Support for custom voice samples in organized subdirectories

### 🎯 Usage

```bash
# Basic usage
python main.py

# With load-on-demand mode (faster startup)
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
We recommend to use NVIDIA Deep Learning Container to manage the CUDA environment. 

### 🔧 Setup Virtual Environment (HIGHLY RECOMMENDED)
Before installing dependencies, it's **highly recommended** to create a virtual environment to avoid conflicts with system packages:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

### 🐳 Docker Installation (Recommended)
1. Launch docker
```bash
# NVIDIA PyTorch Container 24.07 / 24.10 / 24.12 verified. 
# Later versions are also compatible.
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it  nvcr.io/nvidia/pytorch:24.07-py3

## If flash attention is not included in your docker environment, you need to install it manually
## Refer to https://github.com/Dao-AILab/flash-attention for installation instructions
# pip install flash-attn --no-build-isolation
```

2. Install from github
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (required for this application)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install OpenAI (required for AI script generation)
pip install openai

# Install FlashAttention2 (optional, for better performance on CUDA)
# For Windows users, use pre-built wheels:
# pip install flash-attn --no-build-isolation
# Or download from: https://github.com/sunsetcoder/flash-attention-windows

# Install the VibeVoice package
pip install -e .
```

### 💻 Direct Installation (Alternative)
If you prefer not to use Docker, you can install directly on your system:

```bash
# Clone the repository
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice/

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (required for this application)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install OpenAI (required for AI script generation)
pip install openai

# Install FlashAttention2 (optional, for better performance on CUDA)
# For Windows users, use pre-built wheels:
# pip install flash-attn --no-build-isolation
# Or download from: https://github.com/sunsetcoder/flash-attention-windows

# Install dependencies
pip install -e .
```

**Note**: Direct installation requires CUDA-compatible GPU drivers and PyTorch with CUDA support.

### 🔧 Device Compatibility & Fallback Support

VibeVoice now supports multiple device types with automatic fallback mechanisms:

- **CUDA (NVIDIA GPUs)**: Full support with FlashAttention2 for optimal performance
- **Apple Silicon (MPS)**: Native support for M1/M2/M3 Macs using Metal Performance Shaders
- **CPU**: Fallback support for systems without GPU acceleration
- **Windows**: Pre-built FlashAttention2 wheels available for easier installation
- **Robust Fallback**: If FlashAttention2 fails, automatically falls back to SDPA

### 🪟 Windows Quick Start

For Windows users, we provide a convenient batch script to launch VibeVoice:

1. **Install dependencies** (follow installation instructions above)
2. **Configure API keys**: Copy `.env-sample` to `.env` and add your keys
3. **Double-click** `run_vibevoice.bat` to launch

The batch script will:
- ✅ Check for virtual environment
- ✅ Activate the virtual environment automatically  
- ✅ Launch VibeVoice on `http://localhost:7590`
- ✅ Provide helpful error messages if setup is incomplete

**Load-on-Demand Mode**: To use faster startup (loads model when needed), edit `run_vibevoice.bat`:
- Comment out: `python main.py`
- Uncomment: `python main.py --lod`

## Usages

### 🚨 Tips
We observed users may encounter occasional instability when synthesizing Chinese speech. We recommend:

- Using English punctuation even for Chinese text, preferably only commas and periods.
- Using the 7B model variant, which is considerably more stable.

### Usage 1: Launch Gradio demo
```bash
apt update && apt install ffmpeg -y # for demo

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
