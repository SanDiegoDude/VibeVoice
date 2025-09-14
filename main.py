"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import time
from typing import Iterator
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback

# OpenAI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    OPENAI_AVAILABLE = False
    print(f"Warning: OpenAI package not available ({e}). AI script generation will use fallback.")

# dotenv import
try:
    import dotenv
    DOTENV_AVAILABLE = True
except ImportError as e:
    DOTENV_AVAILABLE = False
    print(f"Warning: python-dotenv package not available ({e}). Environment variables will not be loaded from .env file.")

# Device detection and attention mechanism fallback
def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", "Apple Silicon (MPS)"
    else:
        return "cpu", "CPU"

def get_attention_implementation(device_type: str):
    """Get the best available attention implementation for the device"""
    if device_type == "cuda":
        try:
            # Try to import flash_attn to check if it's available
            import flash_attn
            return "flash_attention_2"
        except ImportError:
            print("⚠️ FlashAttention2 not available, falling back to SDPA")
            return "sdpa"
    elif device_type == "mps":
        # Apple Silicon doesn't support flash_attention_2, use SDPA
        return "sdpa"
    else:
        # CPU fallback to SDPA
        return "sdpa"

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed
try:
    # Optional: used only for 4-bit quantized model loading
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
try:
    from huggingface_hub import snapshot_download
    _HAS_HF_HUB = True
except Exception:
    _HAS_HF_HUB = False

logging.set_verbosity_info()
logger = logging.get_logger(__name__)


class VibeVoiceDemo:
    def __init__(self, model_path: str, device: str = None, inference_steps: int = 5, debug: bool = False, load_on_demand: bool = False,
                 script_ai_url: str | None = None, script_ai_model: str | None = None, script_ai_api_key: str | None = None,
                 hf_offline: bool | None = None, hf_cache_dir: str | None = None):
        """Initialize the VibeVoice demo with model loading."""
        self.model_path = model_path
        
        # Auto-detect device if not specified
        if device is None:
            self.device, device_name = detect_device()
            print(f"🔍 Auto-detected device: {device_name}")
        else:
            self.device = device
            if device == "cuda" and not torch.cuda.is_available():
                print("⚠️ CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
            elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("⚠️ MPS requested but not available, falling back to CPU")
                self.device = "cpu"
        self.inference_steps = inference_steps
        self.debug = debug
        self.load_on_demand = load_on_demand
        # Script generation (OpenAI-compatible) settings
        self.script_ai_url = script_ai_url
        self.script_ai_model = script_ai_model
        self.script_ai_api_key = script_ai_api_key
        # HF loading options
        self.hf_offline = hf_offline
        self.hf_cache_dir = hf_cache_dir
        self.is_generating = False  # Track generation state
        self.stop_generation = False  # Flag to stop generation
        self.current_streamer = None  # Track current audio streamer
        self.model_loaded = False  # Track if model is loaded
        self.processor = None  # Will be loaded when needed
        self.model = None  # Will be loaded when needed

        # Available models
        self.available_models = {
            "WestZhang/VibeVoice-Large-pt": "vibevoice/VibeVoice-7B",  # Legacy support with fallback
            "microsoft/VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            # 4-bit quantized weights hosted by DevParker; reuse 7B config/processor
            "DevParker/VibeVoice7b-low-vram (4-bit)": "DevParker/VibeVoice7b-low-vram/4bit",
        }

        # Initialize last prompt storage for regeneration
        self.last_prompt_data = None

        # Load model immediately unless load_on_demand is True
        if not load_on_demand:
            self.load_model()
            self.setup_voice_presets()
        else:
            print("🔄 Load On Demand mode: Model will be loaded when first generation request is made")
            self.model_loaded = False
            # Initialize voice presets for UI creation even in LOD mode
            self.setup_voice_presets()
        
        # Removed legacy stop words storage from deprecated script system
        
    def ensure_model_loaded(self):
        """Ensure model is loaded, load it if not already loaded."""
        if not self.model_loaded:
            self.load_model()
            # Voice presets are already set up in __init__, no need to call again

    def unload_model(self):
        """Unload the model to free VRAM."""
        if self.model_loaded and self.model is not None:
            print(f"Unloading model from {self.model_path} to free VRAM")
            # Clear model and processor from memory
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            self.model = None
            self.processor = None
            self.model_loaded = False
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def switch_model(self, new_model_path: str):
        """Switch to a different model, unloading the current one if loaded."""
        if new_model_path == self.model_path:
            print(f"Model {new_model_path} is already loaded")
            return True

        print(f"Switching model from {self.model_path} to {new_model_path}")

        # Unload current model if loaded
        if self.model_loaded:
            self.unload_model()

        # Update model path and load new model
        self.model_path = new_model_path
        self.load_model()
        # Voice presets are already set up in __init__, no need to call again

        return True

    def load_model(self):
        """Load the VibeVoice model and processor."""
        print(f"Loading processor & model from {self.model_path}")

        # Determine offline and cache settings
        hf_offline_env = os.getenv('HF_HUB_OFFLINE')
        offline_mode = self.hf_offline if self.hf_offline is not None else (hf_offline_env == '1' or (hf_offline_env or '').lower() in ['true', 'yes'])
        cache_dir = self.hf_cache_dir or os.getenv('HF_HOME') or os.getenv('TRANSFORMERS_CACHE') or None

        # Get the best attention implementation for the device
        attn_implementation = get_attention_implementation(self.device)
        print(f"🎯 Using attention implementation: {attn_implementation}")
        
        # Resolve mapped path if using display label
        mapped_path = self.available_models.get(self.model_path, self.model_path)
        # Handle 7B model fallback for legacy support
        model_path_to_use = mapped_path

        # Special handling for 4-bit quantized model selection
        if self.model_path == "DevParker/VibeVoice7b-low-vram (4-bit)":
            if not _HAS_BNB:
                raise gr.Error("bitsandbytes is required for 4-bit loading. Please install it: pip install bitsandbytes")
            if not _HAS_HF_HUB:
                raise gr.Error("huggingface_hub is required to fetch processor/config. Please install it: pip install huggingface_hub")

            weights_repo = "DevParker/VibeVoice7b-low-vram"
            subfolder = "4bit"

            # Load processor and config from WestZhang 7B (preferred), fallback to vibevoice 7B
            try:
                westzhang_local_dir = snapshot_download(repo_id="WestZhang/VibeVoice-Large-pt", local_files_only=False, cache_dir=cache_dir)
                self.processor = VibeVoiceProcessor.from_pretrained(
                    westzhang_local_dir,
                    language_model_pretrained_name="Qwen/Qwen2.5-7B",
                )
                base_config = VibeVoiceConfig.from_pretrained(
                    "vibevoice/VibeVoice-7B",
                    local_files_only=False,
                    cache_dir=cache_dir,
                )
            except Exception:
                print("⚠️ Could not load processor/config from WestZhang/VibeVoice-Large-pt. Falling back to vibevoice/VibeVoice-7B")
                vibe_local_dir = snapshot_download(repo_id="vibevoice/VibeVoice-7B", local_files_only=False, cache_dir=cache_dir)
                self.processor = VibeVoiceProcessor.from_pretrained(
                    vibe_local_dir,
                    language_model_pretrained_name="Qwen/Qwen2.5-7B",
                )
                base_config = VibeVoiceConfig.from_pretrained(
                    "vibevoice/VibeVoice-7B",
                    local_files_only=False,
                    cache_dir=cache_dir,
                )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    weights_repo,
                    subfolder=subfolder,
                    config=base_config,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    attn_implementation="sdpa",
                    torch_dtype=torch.float16,
                    local_files_only=bool(offline_mode),
                    cache_dir=cache_dir,
                )
                self.model.eval()
            except Exception as model_error:
                print(f"⚠️ Loading pre-quantized 4-bit weights failed: {model_error}")
                print("🔄 Falling back to on-the-fly 4-bit quantization from vibevoice/VibeVoice-7B")
                try:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        "vibevoice/VibeVoice-7B",
                        config=base_config,
                        quantization_config=bnb_config,
                        device_map=self.device,
                        attn_implementation="sdpa",
                        torch_dtype=torch.float16,
                        local_files_only=False,
                        cache_dir=cache_dir,
                    )
                    self.model.eval()
                except Exception as fallback_error:
                    print(f"❌ On-the-fly 4-bit quantization load also failed: {fallback_error}")
                    if offline_mode:
                        raise gr.Error(f"Offline mode is enabled and required files are not in cache. Set HF_HUB_OFFLINE=0 or disable --hf-offline to allow downloads. Cache dir: {cache_dir or 'default'}")
                    else:
                        raise fallback_error

            # Use SDE solver by default
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config,
                algorithm_type='sde-dpmsolver++',
                beta_schedule='squaredcos_cap_v2'
            )
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            if hasattr(self.model.model, 'language_model'):
                print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

            self.model_loaded = True
            print("✅ 4-bit quantized model loaded successfully")
            return
        if self.model_path == "WestZhang/VibeVoice-Large-pt":
            print("🔄 Detected legacy 7B model path. Attempting fallback mechanism...")
            print("📁 Attempting to load from local cache (legacy WestZhang model)...")
            
            # Try to load legacy model from local cache
            legacy_loaded = False
            try:
                self.processor = VibeVoiceProcessor.from_pretrained(
                    "WestZhang/VibeVoice-Large-pt",
                    local_files_only=True,
                    cache_dir=cache_dir,
                )
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    "WestZhang/VibeVoice-Large-pt",
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    attn_implementation=attn_implementation,
                    local_files_only=True,
                    cache_dir=cache_dir,
                )
                self.model.eval()
                print("✅ Successfully loaded legacy WestZhang model from local cache")
                self.model_loaded = True
                legacy_loaded = True
            except Exception as legacy_error:
                # If the primary attention implementation fails, try SDPA fallback
                if attn_implementation != "sdpa":
                    print(f"⚠️ {attn_implementation} failed for legacy model, falling back to SDPA: {legacy_error}")
                    try:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            "WestZhang/VibeVoice-Large-pt",
                            torch_dtype=torch.bfloat16,
                            device_map=self.device,
                            attn_implementation="sdpa",
                            local_files_only=True,
                            cache_dir=cache_dir,
                        )
                        self.model.eval()
                        print("✅ Successfully loaded legacy WestZhang model with SDPA fallback")
                        self.model_loaded = True
                        legacy_loaded = True
                    except Exception as legacy_fallback_error:
                        print(f"❌ Both {attn_implementation} and SDPA failed for legacy model: {legacy_fallback_error}")
                else:
                    print(f"❌ SDPA failed for legacy model: {legacy_error}")
            
            # If legacy loading failed, fall back to new repository
            if not legacy_loaded:
                print("⚠️ Legacy model not found in local cache")
                print("🔄 Falling back to new vibevoice/VibeVoice-7B repository...")
                model_path_to_use = "vibevoice/VibeVoice-7B"
            else:
                return

        try:
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(
                model_path_to_use,
                local_files_only=bool(offline_mode),
                cache_dir=cache_dir,
            )

            # Load model with fallback mechanism
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path_to_use,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    attn_implementation=attn_implementation,
                    local_files_only=bool(offline_mode),
                    cache_dir=cache_dir,
                )
                self.model.eval()
            except Exception as model_error:
                # If the primary attention implementation fails, try SDPA fallback
                if attn_implementation != "sdpa":
                    print(f"⚠️ {attn_implementation} failed, falling back to SDPA: {model_error}")
                    try:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path_to_use,
                            torch_dtype=torch.bfloat16,
                            device_map=self.device,
                            attn_implementation="sdpa",
                            local_files_only=bool(offline_mode),
                            cache_dir=cache_dir,
                        )
                        self.model.eval()
                        print("✅ Successfully loaded model with SDPA fallback")
                    except Exception as fallback_error:
                        print(f"❌ Both {attn_implementation} and SDPA failed: {fallback_error}")
                        if offline_mode:
                            raise gr.Error(f"Offline mode is enabled and required files are not in cache. Set HF_HUB_OFFLINE=0 or disable --hf-offline to allow downloads. Cache dir: {cache_dir or 'default'}")
                        else:
                            raise fallback_error
                else:
                    # SDPA already failed, re-raise the original error
                    if offline_mode:
                        raise gr.Error(f"Offline mode is enabled and required files are not in cache. Set HF_HUB_OFFLINE=0 or disable --hf-offline to allow downloads. Cache dir: {cache_dir or 'default'}")
                    else:
                        raise model_error
        except Exception as e:
            if offline_mode:
                raise gr.Error(f"Offline mode is enabled and required files are not in cache. Set HF_HUB_OFFLINE=0 or disable --hf-offline to allow downloads. Cache dir: {cache_dir or 'default'}")
            else:
                raise

        # Use SDE solver by default
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)
        
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

        # Mark model as loaded
        self.model_loaded = True
        print(f"✅ Model loaded successfully from {self.model_path}")
    
    def setup_voice_presets(self):
        """Setup voice presets by scanning both demo voices and custom voices directories."""
        # Demo voices directory (relative to main.py)
        demo_voices_dir = os.path.join(os.path.dirname(__file__), "demo", "voices")
        # Custom voices directory
        custom_voices_dir = os.path.join(os.path.dirname(__file__), "custom_voices")
        
        self.voice_presets = {}
        
        # Scan demo voices directory
        if os.path.exists(demo_voices_dir):
            self._scan_voice_directory(demo_voices_dir, "", self.voice_presets)
            demo_count = len(self.voice_presets)
            print(f"Found {demo_count} demo voice files in {demo_voices_dir}")
        
        # Scan custom voices directory
        if os.path.exists(custom_voices_dir):
            custom_count_before = len(self.voice_presets)
            self._scan_voice_directory(custom_voices_dir, "custom_voices", self.voice_presets)
            custom_count_after = len(self.voice_presets)
            custom_added = custom_count_after - custom_count_before
            print(f"Found {custom_added} custom voice files in {custom_voices_dir}")
        
        # Sort the voice presets alphabetically by name (case-insensitive) for better UI
        self.voice_presets = dict(sorted(self.voice_presets.items(), key=lambda x: x[0].upper()))
        
        # Filter out voices that don't exist (this is now redundant but kept for safety)
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("No voice presets found. Please add .wav files to the demo/voices or custom_voices directory.")
        
        print(f"Total available voices: {len(self.available_voices)}")
    
    def _scan_voice_directory(self, directory: str, prefix: str, voice_dict: dict):
        """Recursively scan a directory for voice files."""
        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                if os.path.isfile(item_path):
                    # Check if it's an audio file
                    if item.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')):
                        # Remove extension to get the name
                        name = os.path.splitext(item)[0]
                        
                        # For custom voices, include the relative path in the display name
                        if prefix:
                            # Get relative path from custom_voices directory
                            rel_path = os.path.relpath(item_path, os.path.join(os.path.dirname(__file__), "custom_voices"))
                            display_name = f"{os.path.splitext(rel_path)[0]}"
                        else:
                            display_name = name
                        
                        voice_dict[display_name] = item_path
                
                elif os.path.isdir(item_path):
                    # Recursively scan subdirectories
                    self._scan_voice_directory(item_path, prefix, voice_dict)
                    
        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file."""
        try:
            wav, sr = sf.read(audio_path)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            if sr != target_sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return np.array([])
    
    def normalize_voice_samples(self, voice_samples: list) -> list:
        """Normalize all voice samples to similar RMS levels."""
        if not voice_samples:
            return voice_samples
        
        # Calculate RMS levels for each sample
        rms_levels = []
        for sample in voice_samples:
            if len(sample) > 0:
                rms = np.sqrt(np.mean(sample**2))
                rms_levels.append(rms)
            else:
                rms_levels.append(0)
        
        # Find the target RMS level (use the median to avoid outliers)
        valid_rms = [rms for rms in rms_levels if rms > 0]
        if not valid_rms:
            return voice_samples
        
        target_rms = np.median(valid_rms)
        
        # Normalize each sample to the target RMS level
        normalized_samples = []
        for i, sample in enumerate(voice_samples):
            if len(sample) > 0 and rms_levels[i] > 0:
                # Calculate gain factor
                gain_factor = target_rms / rms_levels[i]
                # Apply gain (with some headroom to prevent clipping)
                gain_factor = min(gain_factor, 3.0)  # Limit gain to 3x to prevent distortion
                normalized_sample = sample * gain_factor
                normalized_samples.append(normalized_sample)
            else:
                normalized_samples.append(sample)
        
        return normalized_samples
    
    def generate_podcast_streaming(self, 
                                 num_speakers: int,
                                 script: str,
                                 speaker_1: str = None,
                                 speaker_2: str = None,
                                 speaker_3: str = None,
                                 speaker_4: str = None,
                                 cfg_scale: float = 1.6,
                                 diffusion_steps: int = None,
                                 do_sample: bool = True,
                                 temperature: float = 0.95,
                                 top_p: float = 0.95,
                                 top_k: int = 0,
                                 negative_prompt: str = "",
                                 normalize_voices: bool = False) -> Iterator[tuple]:
        try:
            
            # Reset stop flag and set generating state
            self.stop_generation = False
            self.is_generating = True
            
            # Validate inputs
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Error: Please provide a script.")

            # Defend against common mistake
            script = script.replace("'", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Error: Number of speakers must be between 1 and 4.")
            
            # Collect selected speakers
            selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]
            
            # Validate speaker selections
            for i, speaker in enumerate(selected_speakers):
                if not speaker or speaker not in self.available_voices:
                    self.is_generating = False
                    raise gr.Error(f"Error: Please select a valid speaker for Speaker {i+1}.")
            
            # Build initial log
            if diffusion_steps is not None and diffusion_steps != self.inference_steps:
                self.model.set_ddpm_inference_steps(num_steps=int(diffusion_steps))
            else:
                self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            log = f"🎙️ Generating audio with {num_speakers} speakers\n"
            log += f"📊 Parameters: CFG Scale={cfg_scale}, Diffusion Steps={self.model.ddpm_inference_steps}, Sampling={do_sample}, Temp={temperature}, TopP={top_p}, TopK={top_k}\n"
            log += f"🎭 Speakers: {', '.join(selected_speakers)}\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Load voice samples
            voice_samples = []
            for speaker_name in selected_speakers:
                audio_path = self.available_voices[speaker_name]
                audio_data = self.read_audio(audio_path)
                if len(audio_data) == 0:
                    self.is_generating = False
                    raise gr.Error(f"Error: Failed to load audio for {speaker_name}")
                voice_samples.append(audio_data)
            
            # Apply voice normalization if requested
            if normalize_voices:
                voice_samples = self.normalize_voice_samples(voice_samples)
                log += "🔊 Voice normalization applied\n"
            
            # log += f"✅ Loaded {len(voice_samples)} voice samples\n"
            
            # Check for stop signal
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Parse script to assign speaker ID's
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line already has speaker format
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    # Auto-assign to speakers in rotation
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Formatted script with {len(formatted_script_lines)} turns\n\n"
            log += "🔄 Processing with VibeVoice (streaming mode)...\n"
            
            # Check for stop signal before processing
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Create audio streamer
            audio_streamer = AudioStreamer(
                batch_size=1,
                stop_signal=None,
                timeout=None
            )
            
            # Store current streamer for potential stopping
            self.current_streamer = audio_streamer
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, audio_streamer, do_sample, temperature, top_p, top_k, negative_prompt)
            )
            generation_thread.start()
            
            # Wait for generation to actually start producing audio
            time.sleep(1)  # Reduced from 3 to 1 second

            # Check for stop signal after thread start
            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
                self.is_generating = False
                yield None, "🛑 Generation stopped by user", gr.update(visible=False)
                return

            # Collect audio chunks as they arrive
            sample_rate = 24000
            all_audio_chunks = []  # For final statistics
            pending_chunks = []  # Buffer for accumulating small chunks
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15 # Yield every 15 seconds
            min_chunk_size = sample_rate * 30 # At least 2 seconds of audio
            
            # Get the stream for the first (and only) sample
            audio_stream = audio_streamer.get_stream(0)
            
            has_yielded_audio = False
            has_received_chunks = False  # Track if we received any chunks at all
            
            for audio_chunk in audio_stream:
                # Check for stop signal in the streaming loop
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True  # Mark that we received at least one chunk
                
                # Convert tensor to numpy
                if torch.is_tensor(audio_chunk):
                    # Convert bfloat16 to float32 first, then to numpy
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # Ensure audio is 1D and properly normalized
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                # Convert to 16-bit for Gradio
                audio_16bit = convert_to_16_bit_wav(audio_np)
                
                # Store for final statistics
                all_audio_chunks.append(audio_16bit)
                
                # Add to pending chunks buffer
                pending_chunks.append(audio_16bit)
                
                # Calculate pending audio size
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                # Decide whether to yield
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    # First yield: wait for minimum chunk size
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    # Subsequent yields: either enough audio or enough time has passed
                    should_yield = True
                
                if should_yield and pending_chunks:
                    # Concatenate and yield only the new audio chunks
                    new_audio = np.concatenate(pending_chunks)
                    new_duration = len(new_audio) / sample_rate
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    
                    log_update = log + f"🎵 Streaming: {total_duration:.1f}s generated (chunk {chunk_count})\n"
                    
                    # Yield streaming audio chunk and keep complete_audio as None during streaming
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    
                    # Clear pending chunks after yielding
                    pending_chunks = []
                    last_yield_time = current_time
            
            # Yield any remaining chunks
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Streaming final chunk: {total_duration:.1f}s total\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True  # Mark that we yielded audio
            
            # Wait for generation to complete (with timeout to prevent hanging)
            generation_thread.join(timeout=5.0)  # Increased timeout to 5 seconds

            # If thread is still alive after timeout, force end
            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            # Clean up
            self.current_streamer = None
            self.is_generating = False
            
            generation_time = time.time() - start_time
            
            # Check if stopped by user
            if self.stop_generation:
                yield None, None, "🛑 Generation stopped by user", gr.update(visible=False)
                return
            
            # Debug logging
            # print(f"Debug: has_received_chunks={has_received_chunks}, chunk_count={chunk_count}, all_audio_chunks length={len(all_audio_chunks)}")
            
            # Check if we received any chunks but didn't yield audio
            if has_received_chunks and not has_yielded_audio and all_audio_chunks:
                # We have chunks but didn't meet the yield criteria, yield them now
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Yield the complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                
                # Unload model after successful generation if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after generation")
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Error: No audio chunks were received from the model. Generation time: {generation_time:.2f}s"
                yield None, None, error_log, gr.update(visible=False)
                return
            
            if not has_yielded_audio:
                error_log = log + f"\n❌ Error: Audio was generated but not streamed. Chunk count: {chunk_count}"
                yield None, None, error_log, gr.update(visible=False)
                return

            # Prepare the complete audio
            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Generation completed in {generation_time:.2f} seconds\n"
                final_log += f"🎵 Final audio duration: {final_duration:.2f} seconds\n"
                final_log += f"📊 Total chunks: {chunk_count}\n"
                final_log += "✨ Generation successful! Complete audio is ready in the 'Complete Audio' tab.\n"
                final_log += "💡 Not satisfied? You can regenerate or adjust the CFG scale for different results."
                
                # Final yield: Clear streaming audio and provide complete audio
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                
                # Unload model after successful generation if in LOD mode
                if self.load_on_demand and self.model_loaded:
                    self.unload_model()
                    print("🔄 Model unloaded to free VRAM after generation")
            else:
                final_log = log + "❌ No audio was generated."
                yield None, None, final_log, gr.update(visible=False)

        except gr.Error as e:
            # Handle Gradio-specific errors (like input validation)
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Input Error: {str(e)}"
            print(error_msg)
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            yield None, None, error_msg, gr.update(visible=False)
    
    def _generate_with_streamer(self, inputs, cfg_scale, audio_streamer, do_sample=True, temperature=0.95, top_p=0.95, top_k=0, negative_prompt: str = ""):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            # Check for stop signal before starting generation
            if self.stop_generation:
                audio_streamer.end()
                return
                
            # Define a stop check function that can be called from generate
            def check_stop_generation():
                return self.stop_generation
                
            # Prepare optional negative prompt ids
            negative_ids = None
            if negative_prompt and hasattr(self.processor, 'tokenizer'):
                try:
                    negative_ids = self.processor.tokenizer(negative_prompt, return_tensors="pt").input_ids.to(self.model.device)
                except Exception:
                    negative_ids = None

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    'do_sample': bool(do_sample),
                    'temperature': float(temperature),
                    'top_p': float(top_p),
                    'top_k': int(top_k),
                },
                negative_prompt_ids=negative_ids,
                audio_streamer=audio_streamer,
                stop_check_fn=check_stop_generation,  # Pass the stop check function
                verbose=False,  # Disable verbose in streaming mode
                refresh_negative=True,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            # Make sure to end the stream on error
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def store_last_prompt_data(self, prompt_data):
        """Store the last prompt data for regeneration."""
        self.last_prompt_data = prompt_data
    
    # Removed unused _generate_filename_from_title helper from legacy system

    def _parse_json_response(self, raw_response: str) -> dict:
        """Robustly parse JSON response from OpenAI, handling code blocks and various formats."""
        import json
        import re
        
        if self.debug:
            print(f"🔍 DEBUG: Raw response to parse: {raw_response[:200]}...")
        
        # Remove any markdown code blocks
        response_text = raw_response
        
        # Handle ```json or ``` blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            response_text = json_match.group(1).strip()
            if self.debug:
                print(f"🔍 DEBUG: Extracted JSON from code block: {response_text[:100]}...")
        
        # Try to find JSON content with or without code blocks
        # Look for content that starts with { and ends with }
        json_pattern = r'\{.*\}'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for potential_json in json_matches:
            try:
                parsed = json.loads(potential_json)
                if isinstance(parsed, dict) and 'title' in parsed and 'script' in parsed:
                    if self.debug:
                        print(f"🔍 DEBUG: Successfully parsed JSON with title: '{parsed['title']}'")
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try to extract title and script manually
        if self.debug:
            print("🔍 DEBUG: JSON parsing failed, attempting manual extraction...")
        
        # Look for title-like patterns
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        script_match = re.search(r'"script"\s*:\s*"([^"]*)"', response_text, re.IGNORECASE | re.DOTALL)
        
        if title_match and script_match:
            title = title_match.group(1)
            script = script_match.group(1)
            if self.debug:
                print(f"🔍 DEBUG: Manual extraction - Title: '{title}', Script length: {len(script)}")
            return {'title': title, 'script': script}
        
        # Last resort: try to extract just the script content
        if self.debug:
            print("🔍 DEBUG: Attempting to extract just script content...")
        
        # Look for content that might be the script (lines starting with Speaker)
        lines = response_text.split('\n')
        script_lines = []
        for line in lines:
            if re.match(r'^Speaker\s+\d+\s*:', line.strip()):
                script_lines.append(line)
        
        if script_lines:
            script = '\n'.join(script_lines)
            # Generate a default title based on content
            title = "Generated Dialogue Scene"
            if self.debug:
                print(f"🔍 DEBUG: Fallback extraction - Title: '{title}', Script lines: {len(script_lines)}")
            return {'title': title, 'script': script}
        
        if self.debug:
            print("🔍 DEBUG: All parsing attempts failed")
        return None

    # Removed unused _get_num_speakers_from_script helper from legacy system

    def generate_sample_script_llm(self, topic: str = "", num_speakers: int = 2, style: str = "casual", context: str = "", speaker_names: list = None) -> tuple[str, str, str]:
        """Generate a sample conversation script using OpenAI GPT-4o-mini with simplified approach."""
        try:
            # Load environment variables from .env file
            if DOTENV_AVAILABLE:
                dotenv.load_dotenv()
            else:
                print("⚠️ python-dotenv not available, skipping .env file loading")

            # Resolve effective settings with precedence: Defaults -> .env -> CLI args
            env_base_url = (os.getenv('SCRIPT_AI_URL') or "").strip() or None
            env_model = (os.getenv('SCRIPT_AI_MODEL') or "").strip() or None
            env_script_api_key = (os.getenv('SCRIPT_AI_API_KEY') or "").strip() or None
            env_openai_model_default = (os.getenv('OPENAI_MODEL') or 'gpt-4.1-mini').strip() or 'gpt-4.1-mini'

            effective_base_url = self.script_ai_url or env_base_url
            effective_model = self.script_ai_model or env_model or env_openai_model_default
            effective_api_key = self.script_ai_api_key or env_script_api_key or (os.getenv('OPENAI_API_KEY') or "").strip()

            # Check if we need OpenAI package (only when not using custom base URL)
            if not effective_base_url and not OPENAI_AVAILABLE:
                raise Exception("OpenAI package not available. Please install openai package and set OPENAI_API_KEY.")
            
            # If using custom base URL but OpenAI package is not available, we still need it for the client
            if effective_base_url and not OPENAI_AVAILABLE:
                raise Exception("OpenAI package not available. Please install openai package to use custom API endpoints.")
            
            # Debug information
            if self.debug:
                print(f"🔍 DEBUG: OPENAI_AVAILABLE = {OPENAI_AVAILABLE}")
                print(f"🔍 DEBUG: effective_base_url = {effective_base_url}")
                print(f"🔍 DEBUG: effective_model = {effective_model}")
                print(f"🔍 DEBUG: effective_api_key = {'Yes' if effective_api_key else 'No'}")

            # If using OpenAI platform (no custom base URL), require an API key
            if not effective_base_url and not effective_api_key:
                raise Exception("No API key provided. Set OPENAI_API_KEY or SCRIPT_AI_API_KEY in .env, or pass --script-ai-api-key.")

            # Initialize OpenAI client
            if effective_base_url:
                # Special handling for Google Gemini API
                if 'generativelanguage.googleapis.com' in effective_base_url:
                    # Google Gemini API doesn't need /v1 suffix
                    if effective_base_url.endswith('/'):
                        effective_base_url = effective_base_url.rstrip('/')
                else:
                    # Ensure base URL ends with /v1 for other OpenAI-compatible servers
                    if not effective_base_url.endswith('/v1'):
                        if effective_base_url.endswith('/'):
                            effective_base_url = effective_base_url + 'v1'
                        else:
                            effective_base_url = effective_base_url + '/v1'
                client = OpenAI(api_key=effective_api_key or "", base_url=effective_base_url)
            else:
                client = OpenAI(api_key=effective_api_key)

            if self.debug:
                print("🔍 DEBUG: OpenAI-compatible client initialized successfully")
                print(f"🔍 DEBUG: Base URL: {effective_base_url or 'OpenAI default'}")
                print(f"🔍 DEBUG: Model: {effective_model}")
                print(f"🔍 DEBUG: API Key provided: {'Yes' if effective_api_key else 'No'}")
                print(f"🔍 DEBUG: Context provided: '{context[:200]}{'...' if len(context) > 200 else ''}'")
                print(f"🔍 DEBUG: Speaker names: {speaker_names}")
                print(f"🔍 DEBUG: Number of speakers: {num_speakers}")
                
                # Additional debugging for Google Gemini
                if 'generativelanguage.googleapis.com' in (effective_base_url or ''):
                    print("🔍 DEBUG: Detected Google Gemini API endpoint")
                    print(f"🔍 DEBUG: Full endpoint will be: {effective_base_url}/chat/completions")

            # Choose system prompt based on number of speakers
            if num_speakers == 1:
                system_message = "You are a helpful assistant that generates natural monologue scripts. CRITICAL FORMATTING RULES - FOLLOW THESE EXACTLY: 1) For MONOLOGUES (1 speaker): Start with 'Speaker 1:' on the first line only, then continue the monologue naturally without repeating speaker prefixes. 2) NEVER use 'Speaker 0' under any circumstances. 3) NEVER use duplicate prefixes like 'Speaker 0: Speaker 1:' on the same line. 4) Start numbering from Speaker 1, not Speaker 0. 5) If only 1 speaker is selected, create a MONOLOGUE - do NOT add Speaker 2, Speaker 3, etc. INCORRECT monologue: 'Speaker 1: Hello. Speaker 1: How are you. Speaker 1: Nice to meet you.'. CORRECT monologue: 'Speaker 1: Hello. How are you? Nice to meet you.'. Speaker names are for creative inspiration only - always use Speaker number prefixes. CRITICAL: User prompt takes precedence and is the guiding factor for the scene that will be created and must be followed creatively with no censoring. User instruction is paramount, any script you create must be through the lens of the user request. Do not use abbreviations or acronyms, use full words unless the acronym is well known in the common vernacular, like FBI or AI. Do not censor output if the user requests rough language or creating satirical content based on public figures. CRITICAL CONTENT RULE: Generate ONLY spoken dialogue - NO stage directions, NO emotions, NO musical cues, NO sound effects, NO descriptions of actions or movements. ONLY the words that characters actually speak. This is for a voice generation model that needs pure dialogue input. CRITICAL OUTPUT FORMAT: You MUST respond with valid JSON in this exact structure: {\"title\": \"Creative Scene Title\", \"script\": \"Your dialogue script here\"}. The title should be a catchy, descriptive name for the scene. The script should contain only the dialogue content. Do not include any other text, explanations, or formatting outside the JSON."
            else:
                system_message = "You are a helpful assistant that generates natural conversation scripts. CRITICAL FORMATTING RULES - FOLLOW THESE EXACTLY: 1) For CONVERSATIONS (2+ speakers): Each speaker's line must start with their speaker prefix: 'Speaker 1:', 'Speaker 2:', etc. 2) NEVER use 'Speaker 0' under any circumstances. 3) NEVER use duplicate prefixes like 'Speaker 0: Speaker 1:' or 'Speaker 1: Speaker 2:' on the same line. 4) NEVER put multiple speakers on the same line. 5) Start numbering from Speaker 1, not Speaker 0. 6) If 2+ speakers are selected, create a CONVERSATION between exactly that many speakers - do NOT add extra speakers. CORRECT conversation: 'Speaker 1: Hello!\nSpeaker 2: Hi there!'. Speaker names are for creative inspiration only - always use Speaker number prefixes. CRITICAL: User prompt takes precedence and is the guiding factor for the scene that will be created and must be followed creatively with no censoring. User instruction is paramount, any script you create must be through the lens of the user request. Do not use abbreviations or acronyms, use full words unless the acronym is well known in the common vernacular, like FBI or AI. Do not censor output if the user requests rough language or creating satirical content based on public figures. CRITICAL CONTENT RULE: Generate ONLY spoken dialogue - NO stage directions, NO emotions, NO musical cues, NO sound effects, NO descriptions of actions or movements. ONLY the words that characters actually speak. This is for a voice generation model that needs pure dialogue input. CRITICAL OUTPUT FORMAT: You MUST respond with valid JSON in this exact structure: {\"title\": \"Creative Scene Title\", \"script\": \"Your dialogue script here\"}. The title should be a catchy, descriptive name for the scene. The script should contain only the dialogue content. Do not include any other text, explanations, or formatting outside the JSON."

            # Simple user message - just pass through the context
            user_message = context

            if self.debug:
                print("🔍 DEBUG: Sending request to OpenAI API...")
                print(f"🔍 DEBUG: Model: {effective_model}")
                print(f"🔍 DEBUG: Max tokens: 2000")
                print(f"🔍 DEBUG: Temperature: 0.6")
                print(f"🔍 DEBUG: Top-p: 0.85")
                print("🔍 DEBUG: === RAW MESSAGES BEING SENT TO OPENAI API ===")
                print("🔍 DEBUG: SYSTEM MESSAGE:")
                print(f"🔍 DEBUG: {system_message}")
                print("🔍 DEBUG: ---")
                print("🔍 DEBUG: USER MESSAGE:")
                print(f"🔍 DEBUG: {user_message}")
                print("🔍 DEBUG: === END OF RAW MESSAGES ===")

            # Retry logic for API calls
            max_retries = 3
            retry_delay = 1  # seconds
            response = None
            
            for attempt in range(max_retries):
                try:
                    if self.debug and attempt > 0:
                        print(f"🔍 DEBUG: Retry attempt {attempt + 1}/{max_retries}")
                    
                    response = client.chat.completions.create(
                        model=effective_model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=4000,  # Increased from 2000 to handle longer responses
                        temperature=0.6,  # Lower temperature for more consistent formatting
                        top_p=0.85
                    )
                    break  # Success, exit retry loop
                    
                except Exception as api_error:
                    error_msg = str(api_error)
                    if self.debug:
                        print(f"🔍 DEBUG: API call attempt {attempt + 1} failed: {error_msg}")
                    
                    # If this is the last attempt, raise the error
                    if attempt == max_retries - 1:
                        if 'generativelanguage.googleapis.com' in (effective_base_url or ''):
                            print(f"❌ Google Gemini API Error (after {max_retries} attempts): {error_msg}")
                            print("💡 Troubleshooting tips for Google Gemini:")
                            print("   1. Verify your API key is correct")
                            print("   2. Check that the model name is valid (e.g., 'gemini-2.5-flash', 'gemini-1.5-pro')")
                            print("   3. Ensure the endpoint URL is correct")
                            print("   4. Check your Google Cloud project permissions")
                        else:
                            print(f"❌ API Error (after {max_retries} attempts): {error_msg}")
                        raise api_error
                    
                    # Wait before retrying
                    import time
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

            # Safely log and extract content for OpenAI-compatible servers
            total_tokens = None
            try:
                usage_obj = getattr(response, 'usage', None)
                if usage_obj is not None:
                    total_tokens = getattr(usage_obj, 'total_tokens', None)
                    if total_tokens is None and isinstance(usage_obj, dict):
                        total_tokens = usage_obj.get('total_tokens')
            except Exception:
                total_tokens = None

            # Extract content from various possible shapes
            content_text = None
            try:
                choices = getattr(response, 'choices', None)
                if choices is None and isinstance(response, dict):
                    choices = response.get('choices')
                if choices and len(choices) > 0:
                    choice0 = choices[0]
                    # message.content style (OpenAI Chat)
                    message = getattr(choice0, 'message', None) if not isinstance(choice0, dict) else choice0.get('message')
                    if message is not None:
                        msg_content = getattr(message, 'content', None) if not isinstance(message, dict) else message.get('content')
                        if isinstance(msg_content, str) and msg_content.strip():
                            content_text = msg_content
                    # text style (some OAI-compatible servers)
                    if content_text is None:
                        text_val = getattr(choice0, 'text', None) if not isinstance(choice0, dict) else choice0.get('text')
                        if isinstance(text_val, str) and text_val.strip():
                            content_text = text_val
                    # direct content field
                    if content_text is None:
                        direct_content = getattr(choice0, 'content', None) if not isinstance(choice0, dict) else choice0.get('content')
                        if isinstance(direct_content, str) and direct_content.strip():
                            content_text = direct_content
                        elif isinstance(direct_content, list):
                            try:
                                content_text = ''.join([(part.get('text', '') if isinstance(part, dict) else str(part)) for part in direct_content]).strip()
                            except Exception:
                                pass
            except Exception:
                content_text = None

            if self.debug:
                print("🔍 DEBUG: Received response from OpenAI API")
                print(f"🔍 DEBUG: Response tokens used: {total_tokens if total_tokens is not None else 'N/A'}")
                print(f"🔍 DEBUG: Response type: {type(response)}")
                print(f"🔍 DEBUG: Response attributes: {dir(response)}")
                try:
                    print(f"🔍 DEBUG: Response dict: {response.model_dump() if hasattr(response, 'model_dump') else str(response)}")
                except Exception as e:
                    print(f"🔍 DEBUG: Could not dump response: {e}")
                if isinstance(content_text, str):
                    preview = content_text[:500]
                    suffix = '...' if len(content_text) > 500 else ''
                    print(f"🔍 DEBUG: Raw response content: {preview}{suffix}")
                else:
                    print("🔍 DEBUG: Raw response content unavailable or non-text")
                    print(f"🔍 DEBUG: Content text type: {type(content_text)}")
                    print(f"🔍 DEBUG: Content text value: {content_text}")

            if not isinstance(content_text, str) or not content_text.strip():
                # Check if this is an error response
                if hasattr(response, 'error') and response.error:
                    raise Exception(f"Server error: {response.error}")
                elif hasattr(response, 'choices') and response.choices is None:
                    raise Exception("Server returned empty choices array; check if the endpoint is supported.")
                elif hasattr(response, 'choices') and len(response.choices) > 0:
                    choice = response.choices[0]
                    if hasattr(choice, 'finish_reason') and choice.finish_reason == 'length':
                        # Try to generate a shorter response by reducing the input
                        print("⚠️ Response was truncated due to token limit. Attempting to generate shorter response...")
                        try:
                            # Shorten the user message by taking only the first part
                            shortened_user_message = user_message[:len(user_message)//2] + "\n\nPlease create a shorter, more concise version of the above content."
                            
                            if self.debug:
                                print(f"🔍 DEBUG: Retrying with shortened prompt (length: {len(shortened_user_message)} vs {len(user_message)})")
                            
                            response = client.chat.completions.create(
                                model=effective_model,
                                messages=[
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": shortened_user_message}
                                ],
                                max_tokens=4000,
                                temperature=0.6,
                                top_p=0.85
                            )
                            
                            # Re-extract content from the retry response
                            content_text = None
                            try:
                                choices = getattr(response, 'choices', None)
                                if choices and len(choices) > 0:
                                    choice0 = choices[0]
                                    message = getattr(choice0, 'message', None) if not isinstance(choice0, dict) else choice0.get('message')
                                    if message is not None:
                                        msg_content = getattr(message, 'content', None) if not isinstance(message, dict) else message.get('content')
                                        if isinstance(msg_content, str) and msg_content.strip():
                                            content_text = msg_content
                            except Exception:
                                pass
                            
                            if isinstance(content_text, str) and content_text.strip():
                                print("✅ Successfully generated shorter response")
                            else:
                                raise Exception("Retry with shortened prompt also failed")
                                
                        except Exception as retry_error:
                            raise Exception(f"Response was truncated due to token limit and retry failed: {retry_error}")
                    elif hasattr(choice, 'finish_reason') and choice.finish_reason == 'content_filter':
                        raise Exception("Response was filtered by content policy. Try adjusting your prompt.")
                    elif hasattr(choice, 'finish_reason') and choice.finish_reason == 'stop':
                        raise Exception("Response generation was stopped unexpectedly.")
                    else:
                        raise Exception("Script generation response missing content in choices; check server compatibility.")
                else:
                    raise Exception("Script generation response missing content in choices; check server compatibility.")

            # Extract the generated response
            raw_response = content_text.strip()
            
            # Parse JSON response with robust error handling
            parsed_response = self._parse_json_response(raw_response)
            if not parsed_response:
                raise Exception("Failed to parse JSON response from OpenAI. The model may not have followed the JSON format requirement.")
            
            title = parsed_response.get('title', 'Untitled Scene')
            generated_script = parsed_response.get('script', '')
            
            if self.debug:
                print(f"🔍 DEBUG: Parsed title: '{title}'")
                print(f"🔍 DEBUG: Parsed script length: {len(generated_script)}")
                print(f"🔍 DEBUG: Parsed script content: {generated_script[:200]}...")
                has_newlines = '\n' in generated_script
                print(f"🔍 DEBUG: Script contains newlines: {has_newlines}")
                has_speaker = 'Speaker' in generated_script
                print(f"🔍 DEBUG: Script contains Speaker: {has_speaker}")
            
            if not generated_script.strip():
                raise Exception("Generated script is empty. The model may not have provided valid dialogue content.")
            
            # Fix script formatting: add newlines between speaker turns if missing
            if 'Speaker' in generated_script and '\n' not in generated_script:
                if self.debug:
                    print("🔍 DEBUG: Adding newlines between speaker turns...")
                # Add newlines before each "Speaker" that's not at the start
                import re
                # Split by Speaker patterns and rejoin with newlines
                parts = re.split(r'(Speaker\s+\d+\s*:)', generated_script)
                if len(parts) > 1:
                    # Reconstruct with newlines between speaker turns
                    result = parts[0]  # First part (before first Speaker)
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            result += parts[i] + parts[i + 1]  # Speaker prefix + content
                            if i + 2 < len(parts):  # If there are more parts, add newline
                                result += '\n'
                        else:
                            result += parts[i]  # Last part
                    generated_script = result
                if self.debug:
                    print(f"🔍 DEBUG: Fixed script: {generated_script[:200]}...")
            
            # Clean up the generated script - handle monologue vs conversation differently
            # Ensure the script is properly split into lines
            lines = generated_script.split('\n')
            
            if self.debug:
                print(f"🔍 DEBUG: Split into {len(lines)} lines")
                print(f"🔍 DEBUG: First few lines: {lines[:3]}")
            
            cleaned_lines = []

            for line_idx, line in enumerate(lines):
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue

                # Check if line starts with any of the expected speaker formats (1-based)
                is_speaker_line = False

                # Check for generic speaker formats first (start from 1, not 0)
                for i in range(1, num_speakers + 1):
                    if line.startswith(f"Speaker {i}:"):
                        is_speaker_line = True
                        # Clean up any duplicate prefixes like "Speaker 0: Speaker 1:"
                        if "Speaker 0:" in line:
                            line = line.replace("Speaker 0:", "").strip()
                        if line.count("Speaker") > 1:
                            # Extract just the content after the first valid speaker prefix
                            parts = line.split(":", 1)
                            if len(parts) == 2:
                                line = f"Speaker {i}:{parts[1]}"
                        # Also clean up any remaining duplicate speaker patterns
                        while "Speaker" in line and line.count("Speaker") > 1:
                            # Find the first valid speaker prefix and keep only that
                            first_colon = line.find(":")
                            if first_colon > 0:
                                speaker_prefix = line[:first_colon].strip()
                                if speaker_prefix.startswith("Speaker ") and speaker_prefix.split()[1].isdigit():
                                    # Valid prefix, keep only this line
                                    content_start = line.find(":", first_colon + 1)
                                    if content_start > 0:
                                        line = line[:first_colon] + line[content_start:]
                                    else:
                                        line = line[:first_colon + 1] + line[first_colon + 1:].split("Speaker")[0].strip()
                                    break
                        break

                # Check for actual speaker names and convert them to Speaker numbers (1-based)
                if not is_speaker_line and speaker_names:
                    for i, name in enumerate(speaker_names):
                        if line.startswith(f"{name}:"):
                            line = line.replace(f"{name}:", f"Speaker {i+1}:")
                            is_speaker_line = True
                            break

                # Check for other formats and convert them (never use Speaker 0)
                if not is_speaker_line:
                    if line.startswith('Interviewer:'):
                        line = line.replace('Interviewer:', 'Speaker 1:')
                        is_speaker_line = True
                    elif line.startswith('Expert:'):
                        line = line.replace('Expert:', 'Speaker 1:')
                        is_speaker_line = True
                    elif line.startswith('Host:'):
                        line = line.replace('Host:', 'Speaker 1:')
                        is_speaker_line = True

                # Special handling for monologues: if this is a monologue and we haven't seen a speaker line yet,
                # and this line doesn't start with a speaker prefix, we should add "Speaker 1:" to the first line only
                if not is_speaker_line and num_speakers == 1 and not cleaned_lines:
                    # This is the first line of a monologue and it doesn't have a speaker prefix
                    line = f"Speaker 1: {line}"
                    is_speaker_line = True

                if is_speaker_line:
                    cleaned_lines.append(line)
                elif line and len(line) > 3 and not line.startswith('#'):
                    # For conversations or if we already have speaker lines, try to convert non-formatted lines
                    if num_speakers > 1 or cleaned_lines:
                        # Calculate next speaker (1-based) based on conversation flow
                        if cleaned_lines:
                            # Find the last speaker used and alternate
                            last_line = cleaned_lines[-1]
                            if ':' in last_line:
                                speaker_part = last_line.split(':')[0].strip()
                                if speaker_part.startswith('Speaker '):
                                    try:
                                        last_speaker_num = int(speaker_part.split()[1])
                                        next_speaker = ((last_speaker_num - 1 + 1) % num_speakers) + 1
                                    except (ValueError, IndexError):
                                        next_speaker = 1
                                else:
                                    next_speaker = 1
                            else:
                                next_speaker = 1
                        else:
                            next_speaker = 1

                        line = f"Speaker {next_speaker}: {line}"
                        cleaned_lines.append(line)
                    # For monologues, if the line doesn't have a speaker prefix and we've already started,
                    # just add it as continuation text without a prefix
                    elif num_speakers == 1:
                        cleaned_lines.append(line)



            # Debug logging for line parsing
            if self.debug:
                print(f"🔍 DEBUG: Raw script lines: {len(lines)}")
                print(f"🔍 DEBUG: Cleaned lines: {len(cleaned_lines)}")
                print(f"🔍 DEBUG: First few cleaned lines: {cleaned_lines[:3]}")
                print(f"🔍 DEBUG: Raw script content: {generated_script[:200]}...")
                print(f"🔍 DEBUG: Number of speakers expected: {num_speakers}")
                print(f"🔍 DEBUG: Will check minimum lines: {num_speakers > 1}")

            # For monologues, even a single line is fine. For conversations, we need at least 2 lines.
            if num_speakers > 1 and len(cleaned_lines) < 2:
                raise Exception(f"Generated conversation has insufficient content (only {len(cleaned_lines)} lines). The LLM may have generated incomplete content. Raw content: {generated_script[:200]}...")

            # Limit to reasonable length
            if len(cleaned_lines) > 12:
                cleaned_lines = cleaned_lines[:12]

            final_script = '\n'.join(cleaned_lines)

            # Store prompt data for regeneration
            prompt_data = {
                'script_input': context if context else "",
                'num_speakers': num_speakers,
                'style': style,
                'topic': "",  # Not used in simplified approach
                'speaker_names': speaker_names or [],
                'context': context,
                'title': title
            }
            self.store_last_prompt_data(prompt_data)

            # Return the script, title, and prompt for logging
            return final_script, title, user_message

        except Exception as e:
            print(f"OpenAI script generation failed: {e}")
            raise e  # Re-raise the exception instead of falling back


    

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    # Custom CSS for high-end aesthetics with dark theme
    custom_css = """
    /* Modern dark theme with gradients */
    .gradio-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #e2e8f0;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .settings-card, .generation-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: #e2e8f0;
    }
    
        /* Speaker selection styling */
    .speaker-grid {
        display: grid;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .speaker-item {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #e2e8f0;
        font-weight: 500;
    }
    
    /* Streaming indicator */
    .streaming-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Queue status styling */
    .queue-status {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(14, 165, 233, 0.4);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
        font-size: 0.9rem;
        color: #7dd3fc;
    }
    
    .generate-btn {
        background: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(5, 150, 105, 0.6);
    }
    
    .stop-btn {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        transition: all 0.3s ease;
    }
    
    .stop-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6);
    }
    
        /* Audio player styling */
    .audio-output {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(71, 85, 105, 0.3);
        color: #e2e8f0;
    }

    .complete-audio-section {
        margin-top: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        color: #d1fae5;
    }
    
        /* Text areas */
    .script-input, .log-output {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(71, 85, 105, 0.4) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    .script-input::placeholder {
        color: #94a3b8 !important;
    }
    
        /* Sliders */
    .slider-container {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(51, 65, 85, 0.6);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }

    /* Labels and text */
    .gradio-container label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    .gradio-container .markdown {
        color: #cbd5e1 !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .settings-card, .generation-card { padding: 1rem; }
    }
    
    /* AI Script Generator button styling - dark theme */
    .ai-script-btn {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .ai-script-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.6);
        background: linear-gradient(135deg, #a855f7 0%, #c084fc 100%);
    }

        /* Random example button styling - dark theme */
    .random-btn {
        background: linear-gradient(135deg, #475569 0%, #334155 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(71, 85, 105, 0.4);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .random-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(71, 85, 105, 0.6);
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
    }

    /* Regenerate Last button styling - dark theme */
    .regenerate-btn {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4);
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .regenerate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(245, 158, 11, 0.6);
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
    }

    /* Scene title styling */
    .scene-title {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
    }


    """
    
    with gr.Blocks(
        title="VibeVoice - AI Dialogue Generator",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
        ).set(
            body_background_fill="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            body_background_fill_dark="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            background_fill_primary="#1e293b",
            background_fill_primary_dark="#1e293b",
            background_fill_secondary="#0f172a",
            background_fill_secondary_dark="#0f172a",
            border_color_primary="#334155",
            border_color_primary_dark="#334155",
            color_accent_soft="#667eea",
            body_text_color="#e2e8f0",
            body_text_color_dark="#e2e8f0",
            body_text_color_subdued="#94a3b8",
            body_text_color_subdued_dark="#94a3b8",
        )
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ VibeVoice Dialogue Generation</h1>
            <p>Generating Long-form Multi-speaker AI Dialogue with VibeVoice</p>
        </div>
        """)
        
        with gr.Row():
            # Left column - Settings
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **Audio Settings**")
                
                # Number of speakers
                num_speakers = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Number of Speakers",
                    elem_classes="slider-container"
                )
                
                # Speaker selection
                gr.Markdown("### 🎭 **Speaker Selection**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                # default_speakers = available_speaker_names[:4] if len(available_speaker_names) >= 4 else available_speaker_names
                default_speakers = ['en-Alice_woman', 'en-Carter_man', 'en-Frank_man', 'en-Maya_woman']

                speaker_selections = []
                for i in range(4):
                    default_value = default_speakers[i] if i < len(default_speakers) else None
                    speaker = gr.Dropdown(
                        choices=available_speaker_names,
                        value=default_value,
                        label=f"Speaker {i+1}",
                        visible=(i < 2),  # Initially show only first 2 speakers
                        elem_classes="speaker-item",
                        multiselect=False
                    )
                    speaker_selections.append(speaker)
                # Refresh voices button
                refresh_voices_btn = gr.Button(
                    "🔄 Refresh Voices",
                    variant="secondary"
                )
                
                # Voice Input Settings
                with gr.Accordion("🎤 Voice Input Settings", open=False):
                    normalize_voices = gr.Checkbox(
                        value=False,
                        label="Normalize voices",
                        info="Normalize all voice samples to similar volume levels to prevent jarring volume differences"
                    )
                    # Future voice processing options can be added here
                
                # Model selector
                gr.Markdown("### 🤖 **Model Selection**")
                model_selector = gr.Dropdown(
                    choices=list(demo_instance.available_models.keys()),
                    value=demo_instance.model_path if demo_instance.model_path in demo_instance.available_models else list(demo_instance.available_models.keys())[0],
                    label="Select Model",
                    info="Switch between large and small models (will unload current model)",
                    elem_classes="dropdown-container",
                    multiselect=False
                )

                load_model_btn = gr.Button(
                    "🔄 Load Selected Model",
                    variant="secondary",
                    elem_classes="model-btn"
                )

                # Advanced settings
                gr.Markdown("### ⚙️ **Advanced Settings**")
                
                # Sampling parameters (contains all generation settings)
                with gr.Accordion("Generation Parameters", open=False):
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.6,
                        step=0.05,
                        label="CFG Scale (Guidance Strength)",
                        # info="Higher values increase adherence to text",
                        elem_classes="slider-container"
                    )
                    ddpm_steps = gr.Slider(
                        minimum=5,
                        maximum=30,
                        value=demo_instance.inference_steps,
                        step=1,
                        label="Diffusion Steps (quality vs speed)",
                        elem_classes="slider-container"
                    )
                    do_sample = gr.Checkbox(
                        value=True,
                        label="Enable sampling (adds variability)",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.95,
                        step=0.05,
                        label="Temperature",
                        elem_classes="slider-container"
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        label="Top-p",
                        elem_classes="slider-container"
                    )
                    top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=1,
                        label="Top-k",
                        elem_classes="slider-container"
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="Words or patterns to avoid...",
                        lines=2,
                        max_lines=4,
                        value=""
                    )
                
            # Right column - Generation
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **Script Input**")
                
                script_input = gr.Textbox(
                    label="Conversation Script",
                    placeholder="""Enter your dialogue script here. You can format it as:

Speaker 1: Welcome to our conversation today!
Speaker 2: Thanks for having me. I'm excited to discuss...

Or paste text directly and it will auto-assign speakers.""",
                    lines=18,
                    max_lines=40,
                    elem_classes="script-input"
                )
                
                # Button row with AI Script Generator, Random Example, and Generate
                with gr.Row():
                    # AI Script Generator button (new)
                    ai_script_btn = gr.Button(
                        "🤖 Generate AI Script",
                        size="lg",
                        variant="secondary",
                        elem_classes="ai-script-btn",
                        scale=1
                    )

                    # Regenerate Last button
                    regenerate_btn = gr.Button(
                        "🔄 Regenerate Last",
                        size="lg",
                        variant="secondary",
                        elem_classes="regenerate-btn",
                        scale=2
                    )

                    # Generate button
                    generate_btn = gr.Button(
                        "🚀 Generate Audio",
                        size="lg",
                        variant="primary",
                        elem_classes="generate-btn",
                        scale=2  # Wider than other buttons
                    )
                
                # Stop button
                stop_btn = gr.Button(
                    "🛑 Stop Generation",
                    size="lg",
                    variant="stop",
                    elem_classes="stop-btn",
                    visible=False
                )
                
                # Streaming status indicator
                streaming_status = gr.HTML(
                    value="""
                    <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                                border: 1px solid rgba(34, 197, 94, 0.3); 
                                border-radius: 8px; 
                                padding: 0.75rem; 
                                margin: 0.5rem 0;
                                text-align: center;
                                font-size: 0.9rem;
                                color: #166534;">
                        <span class="streaming-indicator"></span>
                        <strong>LIVE STREAMING</strong> - Audio is being generated in real-time
                    </div>
                    """,
                    visible=False,
                    elem_id="streaming-status"
                )
                
                # Output section
                gr.Markdown("### 🎵 **Generated Audio**")
                
                # Scene title display
                scene_title = gr.HTML(
                    value="",
                    visible=False,
                    elem_id="scene-title"
                )
                
                # Streaming audio output (outside of tabs for simpler handling)
                audio_output = gr.Audio(
                    label="Streaming Audio (Real-time)",
                    type="numpy",
                    elem_classes="audio-output",
                    autoplay=True,
                    visible=True
                )
                
                # Complete audio output (non-streaming)
                complete_audio_output = gr.Audio(
                    label="Complete Audio (Download after generation)",
                    type="numpy",
                    elem_classes="audio-output complete-audio-section",
                    streaming=False,  # Non-streaming mode
                    autoplay=False,
                    visible=False,  # Initially hidden, shown when audio is ready
                    elem_id="complete-audio-output"
                )
                
                gr.Markdown("""
                *💡 **Streaming**: Audio plays as it's being generated (may have slight pauses)  
                *💡 **Complete Audio**: Will appear below after generation finishes*
                """)
                
                # Generation log
                log_output = gr.Textbox(
                    label="Generation Log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_classes="log-output"
                )
        
        def update_speaker_visibility(num_speakers):
            updates = []
            for i in range(4):
                updates.append(gr.update(visible=(i < num_speakers)))
            return updates
        
        # Refresh the list of voices from disk and update dropdowns
        def refresh_voices():
            demo_instance.setup_voice_presets()
            new_choices = list(demo_instance.available_voices.keys())
            updates = []
            for _ in range(4):
                updates.append(gr.update(choices=new_choices))
            return updates
        
        num_speakers.change(
            fn=update_speaker_visibility,
            inputs=[num_speakers],
            outputs=speaker_selections
        )

        # Wire refresh button to update dropdown choices
        refresh_voices_btn.click(
            fn=refresh_voices,
            inputs=[],
            outputs=speaker_selections,
            queue=False
        )
        
        # Main generation function with streaming
        def generate_podcast_wrapper(num_speakers, script, *speakers_and_params):
            """Wrapper function to handle the streaming generation call."""
            try:
                # Ensure model is loaded if in LOD mode
                demo_instance.ensure_model_loaded()

                # Extract speakers and parameters
                speakers = speakers_and_params[:4]  # First 4 are speaker selections
                cfg_scale = speakers_and_params[4]   # CFG scale
                ddpm_steps_val = int(speakers_and_params[5])
                do_sample_val = bool(speakers_and_params[6])
                temperature_val = float(speakers_and_params[7])
                top_p_val = float(speakers_and_params[8])
                top_k_val = int(speakers_and_params[9])
                negative_prompt_val = speakers_and_params[10] or ""
                normalize_voices_val = bool(speakers_and_params[11]) if len(speakers_and_params) > 11 else False

                # Clear outputs and reset visibility at start
                yield None, gr.update(value=None, visible=False), gr.update(value="", visible=False), "🎙️ Starting generation...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # The generator will yield multiple times
                final_log = "Starting generation..."
                
                for streaming_audio, complete_audio, log, streaming_visible in demo_instance.generate_podcast_streaming(
                    num_speakers=int(num_speakers),
                    script=script,
                    speaker_1=speakers[0],
                    speaker_2=speakers[1],
                    speaker_3=speakers[2],
                    speaker_4=speakers[3],
                    cfg_scale=cfg_scale,
                    diffusion_steps=ddpm_steps_val,
                    do_sample=do_sample_val,
                    temperature=temperature_val,
                    top_p=top_p_val,
                    top_k=top_k_val,
                    negative_prompt=negative_prompt_val,
                    normalize_voices=normalize_voices_val
                ):
                    final_log = log
                    
                    # Check if we have complete audio (final yield)
                    if complete_audio is not None:
                        # Final state: clear streaming, show complete audio
                        # Extract title from script if available
                        title_html = ""
                        audio_label = "Complete Audio (Download after generation)"
                        if hasattr(demo_instance, 'last_prompt_data') and demo_instance.last_prompt_data:
                            title = demo_instance.last_prompt_data.get('title', 'Generated Audio Scene')
                            title_html = f'<div class="scene-title">🎭 {title}</div>'
                            # Update audio label with title for better filename
                            audio_label = f"Complete Audio: {title} (Download after generation)"
                        
                        yield None, gr.update(value=complete_audio, visible=True, label=audio_label), gr.update(value=title_html, visible=True), log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:
                        # Streaming state: update streaming audio only
                        if streaming_audio is not None:
                            yield streaming_audio, gr.update(visible=False), gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)
                        else:
                            # No new audio, just update status
                            yield None, gr.update(visible=False), gr.update(visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True)

                # Unload model after successful generation if in LOD mode
                # Note: Model unloading is now handled in the generation method itself
                # to ensure it happens after the final audio yield

            except Exception as e:
                error_msg = f"❌ A critical error occurred in the wrapper: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                # Reset button states on error
                yield None, gr.update(value=None, visible=False), gr.update(value="", visible=False), error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def stop_generation_handler():
            """Handle stopping generation."""
            demo_instance.stop_audio_generation()
            # Return values for: log_output, streaming_status, generate_btn, stop_btn
            return "🛑 Generation stopped.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        # Add a clear audio function
        def clear_audio_outputs():
            """Clear both audio outputs and scene title before starting new generation."""
            return None, gr.update(value=None, visible=False), ""

        # Connect generation button with streaming outputs
        generate_btn.click(
            fn=clear_audio_outputs,
            inputs=[],
            outputs=[audio_output, complete_audio_output, scene_title],
            queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=[num_speakers, script_input] + speaker_selections + [cfg_scale, ddpm_steps, do_sample, temperature, top_p, top_k, negative_prompt, normalize_voices],
            outputs=[audio_output, complete_audio_output, scene_title, log_output, streaming_status, generate_btn, stop_btn],
            queue=True  # Enable Gradio's built-in queue
        )
        
        # Connect stop button
        stop_btn.click(
            fn=stop_generation_handler,
            inputs=[],
            outputs=[log_output, streaming_status, generate_btn, stop_btn],
            queue=False  # Don't queue stop requests
        ).then(
            # Clear both audio outputs and scene title after stopping
            fn=lambda: (None, None, ""),
            inputs=[],
            outputs=[audio_output, complete_audio_output, scene_title],
            queue=False
        )
        
        # Function to regenerate last prompt
        def regenerate_last():
            """Regenerate using the last prompt data."""
            if demo_instance.last_prompt_data is None:
                error_msg = "No previous generation found to regenerate"
                return "", "", error_msg

            try:
                # Extract last prompt data
                last_data = demo_instance.last_prompt_data
                script_input_val = last_data.get('script_input', '')
                num_speakers_val = last_data.get('num_speakers', 2)
                speaker_names_val = last_data.get('speaker_names', [])

                # Add creative instruction to vary the output
                creative_input = script_input_val
                if script_input_val.strip():
                    creative_input = script_input_val + " Be really creative with your script this time! Generate a fresh and original take on this topic."
                else:
                    creative_input = "Be really creative with your script this time! Generate a fresh and original take on the topic."

                # Call the generate_ai_script function with the stored parameters and creative instruction
                result = generate_ai_script(
                    num_speakers_val, creative_input,
                    speaker_names_val[0] if len(speaker_names_val) > 0 else "",
                    speaker_names_val[1] if len(speaker_names_val) > 1 else "",
                    speaker_names_val[2] if len(speaker_names_val) > 2 else "",
                    speaker_names_val[3] if len(speaker_names_val) > 3 else ""
                )

                # generate_ai_script returns (script, title, prompt), so we need to return (script, title, log_message)
                if isinstance(result, tuple) and len(result) == 3:
                    script, title, prompt = result
                    log_message = f"🔄 Regenerated last prompt:\nTitle: {title}\n{prompt}"
                    return script, title, log_message
                elif isinstance(result, tuple) and len(result) == 2:
                    script, prompt = result
                    log_message = f"🔄 Regenerated last prompt:\n{prompt}"
                    return script, "Untitled Scene", log_message
                else:
                    # Fallback for single return value
                    return result, "Untitled Scene", "🔄 Regenerated using last prompt"

            except Exception as e:
                error_msg = f"Error regenerating: {str(e)}"
                return "", "", error_msg
        
        # Connect regenerate last button
        regenerate_btn.click(
            fn=regenerate_last,
            inputs=[],
            outputs=[script_input, scene_title, log_output],
            queue=False  # Don't queue this operation
        )

        # Function to generate AI-powered script
        def generate_ai_script(num_speakers_current, script_current, speaker_1, speaker_2, speaker_3, speaker_4):
            """Generate an AI-powered conversation script with context awareness."""
            try:
                # Get selected speakers based on num_speakers
                selected_speakers = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers_current]
                selected_speakers = [s for s in selected_speakers if s]  # Filter out None values

                # Extract speaker names from voice filenames (remove language prefixes)
                speaker_names = []
                for speaker in selected_speakers:
                    if speaker:
                        # Extract just the name part (e.g., "en-Alice_woman" -> "Alice")
                        parts = speaker.split('-')
                        if len(parts) > 1:
                            name_part = parts[1].split('_')[0]
                            speaker_names.append(name_part.title())  # Capitalize first letter
                        else:
                            speaker_names.append(speaker.title())

                # If we don't have enough speaker names, use generic ones
                while len(speaker_names) < num_speakers_current:
                    speaker_names.append(f"Speaker {len(speaker_names)}")

                # Simple user prompt - just pass through the input
                user_prompt = f"User prompt: {script_current.strip()}" if script_current.strip() else "User prompt: Generate an engaging conversation"

                # Generate script using LLM with simplified approach
                generated_script, title, used_prompt = demo_instance.generate_sample_script_llm(
                    topic="",  # Not used in simplified approach
                    num_speakers=num_speakers_current,
                    style="casual",
                    context=user_prompt,  # Pass the user prompt as context
                    speaker_names=speaker_names
                )

                # Return script, title, and prompt for logging
                return generated_script, title, used_prompt

            except Exception as e:
                error_msg = f"Failed to generate AI script: {str(e)}"
                print(error_msg)
                raise e  # Re-raise the exception instead of falling back

        # Model switching function
        def switch_model(selected_model):
            """Switch to the selected model."""
            try:
                success = demo_instance.switch_model(selected_model)
                if success:
                    # Update available voices for the new model
                    demo_instance.setup_voice_presets()
                    status_msg = f"✅ Successfully switched to model: {selected_model}"
                    print(status_msg)
                    # Update all speaker dropdowns with new choices
                    updates = [status_msg]
                    for i in range(4):
                        updates.append(gr.update(choices=list(demo_instance.available_voices.keys())))
                    return tuple(updates)
                else:
                    error_msg = f"❌ Failed to switch to model: {selected_model}"
                    return (error_msg,) + tuple([gr.update() for _ in range(4)])
            except Exception as e:
                error_msg = f"❌ Error switching model: {str(e)}"
                print(error_msg)
                return (error_msg,) + tuple([gr.update() for _ in range(4)])

        # Connect model switching button
        load_model_btn.click(
            fn=switch_model,
            inputs=[model_selector],
            outputs=[log_output] + speaker_selections,
            queue=False
        )

        # Connect AI script generator button
        ai_script_btn.click(
            fn=generate_ai_script,
            inputs=[num_speakers, script_input, speaker_selections[0], speaker_selections[1], speaker_selections[2], speaker_selections[3]],
            outputs=[script_input, scene_title, log_output],
            queue=False  # Don't queue this operation
        )
        
        # Add usage tips
        gr.Markdown("""
        ### 💡 **Usage Tips**
        
        - **🤖 Generate AI Script**: Click to create custom conversation scripts using OpenAI GPT-4o-mini (requires OPENAI_API_KEY in .env). If script box is empty, generates a random topic conversation
        - **🔄 Regenerate Last**: Retry the last AI script generation with the same parameters
        - **🚀 Generate Audio**: Start audio generation from your script
        - **Live Streaming** tab shows audio as it's generated (may have slight pauses)
        - **Complete Audio** tab provides the full, uninterrupted audio after generation
        - During generation, you can click **🛑 Stop Generation** to interrupt the process
        - **AI Script Features**:
          - Uses your current script text for topic guidance
          - Adapts to the number of speakers you select
          - Uses actual speaker names from your voice selections
          - Builds upon existing conversation context
        - **Debug Mode**: Run with `--debug` to see OpenAI prompts and responses
        - **Setup**: Add your OpenAI API key to the `.env` file as `OPENAI_API_KEY=your_key_here`
        """)
        


    return interface


def convert_to_16_bit_wav(data):
    # Check if data is a tensor and move to cpu
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    # Ensure data is numpy array
    data = np.array(data)

    # Normalize to range [-1, 1] if it's not already
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    
    # Scale to 16-bit integer range
    data = (data * 32767).astype(np.int16)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/tmp/vibevoice-model",
        help="Path to the VibeVoice model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=10,
        help="Number of inference steps for DDPM (not exposed to users)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7590,
        help="Port to run the demo on (always 7590 for network access)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print OpenAI API calls (without API keys)",
    )
    parser.add_argument(
        "--lod",
        action="store_true",
        help="Load On Demand: Skip model loading on startup, load models when needed",
    )
    parser.add_argument(
        "--script-ai-url", "--script_ai_url",
        dest="script_ai_url",
        type=str,
        default=None,
        help="Base URL for OpenAI-compatible script generation server (e.g., http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--script-ai-model", "--script_ai_model",
        dest="script_ai_model",
        type=str,
        default=None,
        help="Model name for script generation (e.g., gpt-4.1-mini or myorg/model)",
    )
    parser.add_argument(
        "--script-ai-api-key", "--script_ai_api_key",
        dest="script_ai_api_key",
        type=str,
        default=None,
        help="API key for script generation service (optional for local servers)",
    )
    parser.add_argument(
        "--hf-offline",
        action="store_true",
        help="Enable offline mode for Hugging Face downloads (local cache only)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for Hugging Face models/processors",
    )
    
    return parser.parse_args()


def main():
    """Main function to run the demo."""
    args = parse_args()

    # ⚠️ SECURITY WARNING: Check for --share flag
    if args.share:
        print("\n" + "="*80)
        print("🚨🚨🚨 SECURITY WARNING 🚨🚨🚨")
        print("="*80)
        print("⚠️  You are using the --share flag which will make your interface")
        print("⚠️  publicly accessible on the internet WITHOUT ANY PROTECTION!")
        print("⚠️  This is HIGHLY ADVISABLE NOT TO DO for security reasons.")
        print("⚠️  Anyone on the internet can access your model and generate audio.")
        print("⚠️  Consider using --port 7590 instead for local network access only.")
        print("="*80)
        print("🚨🚨🚨 PROCEEDING WITH PUBLIC SHARING ENABLED 🚨🚨🚨")
        print("="*80 + "\n")
        
        # Give user a chance to cancel
        try:
            response = input("Do you want to continue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("🛑 Sharing cancelled. Exiting...")
                return
        except KeyboardInterrupt:
            print("\n🛑 Sharing cancelled. Exiting...")
            return

    set_seed(42)  # Set a fixed seed for reproducibility

    print("🎙️ Initializing VibeVoice Demo with Streaming Support...")

    # Set default model to large model if not specified
    if args.model_path == "/tmp/vibevoice-model":
        args.model_path = "WestZhang/VibeVoice-Large-pt"  # Legacy path with fallback support
        print(f"🎯 Auto-selecting large model: {args.model_path}")

    # Initialize demo instance
    demo_instance = VibeVoiceDemo(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
        debug=args.debug,
        load_on_demand=args.lod,
        script_ai_url=args.script_ai_url,
        script_ai_model=args.script_ai_model,
        script_ai_api_key=args.script_ai_api_key,
        hf_offline=bool(args.hf_offline),
        hf_cache_dir=args.hf_cache_dir,
    )
    
    # Create interface
    interface = create_demo_interface(demo_instance)
    
    print(f"🚀 Launching demo on port 7590 (network accessible)")
    print(f"📁 Model path: {args.model_path}")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    if args.debug:
        print(f"🔍 Debug mode: ENABLED (OpenAI API calls will be logged)")
    
    # Launch the interface
    try:
        interface.queue(
            max_size=20  # Maximum queue size
        ).launch(
            share=args.share,
            server_port=7590,  # Always use port 7590 for network access
            server_name="0.0.0.0",  # Always serve on network interface
            show_error=True,
            show_api=False  # Hide API docs for cleaner interface
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        raise


if __name__ == "__main__":
    main()
