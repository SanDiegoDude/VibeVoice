"""
Vocal Isolation Module using Mel-Band-Roformer

This module provides vocal isolation functionality for audio preprocessing.
It automatically downloads the Mel-Band-Roformer model from HuggingFace if not present.

Based on: https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model
ComfyUI reference: https://github.com/kijai/ComfyUI-MelBandRoFormer
Model: https://huggingface.co/KimberleyJSN/melbandroformer
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from functools import partial
from tqdm import tqdm

# Model configuration matching ComfyUI's tested settings
MEL_BAND_ROFORMER_CONFIG = {
    "dim": 384,
    "depth": 6,  # ComfyUI uses 6, not 12
    "stereo": True,
    "num_stems": 1,
    "time_transformer_depth": 1,
    "freq_transformer_depth": 1,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,  # ComfyUI uses 0, not 0.1
    "ff_dropout": 0,  # ComfyUI uses 0, not 0.1
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": 44100,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}

# Model download URLs and paths
HUGGINGFACE_MODEL_ID = "KimberleyJSN/melbandroformer"
MODEL_FILENAME = "MelBandRoformer.ckpt"
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "vocal_isolation")


def get_model_path() -> str:
    """Get the path to the vocal isolation model, downloading if necessary."""
    model_dir = Path(DEFAULT_MODEL_DIR)
    model_path = model_dir / MODEL_FILENAME
    
    if not model_path.exists():
        print(f"🔽 Vocal isolation model not found. Downloading from HuggingFace...")
        download_model(model_dir)
    
    return str(model_path)


def download_model(model_dir: Path) -> None:
    """Download the Mel-Band-Roformer model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download the vocal isolation model. "
            "Please install it: pip install huggingface_hub"
        )
    
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {MODEL_FILENAME} from {HUGGINGFACE_MODEL_ID}...")
    print(f"   Target directory: {model_dir}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=HUGGINGFACE_MODEL_ID,
            filename=MODEL_FILENAME,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Model downloaded successfully to: {downloaded_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download vocal isolation model: {e}")


def get_windowing_array(window_size: int, fade_size: int, device: torch.device) -> torch.Tensor:
    """Create windowing array for smooth chunk transitions (from ComfyUI implementation)."""
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


class VocalIsolator:
    """
    High-level interface for vocal isolation using Mel-Band-Roformer.
    
    This class handles model loading, audio preprocessing, and vocal extraction.
    Uses the same processing approach as the ComfyUI implementation for best quality.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        debug: bool = False
    ):
        """
        Initialize the VocalIsolator.
        
        Args:
            model_path: Path to the model checkpoint. If None, auto-downloads.
            device: Device to run inference on ('cuda', 'cpu', 'mps', or None for auto-detect)
            debug: Enable debug logging
        """
        self.debug = debug
        self.model = None
        self.model_sample_rate = 44100  # Mel-Band-Roformer expects 44100Hz
        
        # Processing parameters matching ComfyUI
        self.chunk_size = 352800  # 8 seconds at 44100Hz
        self.num_overlap = 2  # N = 2 in ComfyUI
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Get model path (downloads if necessary)
        self.model_path = model_path if model_path else get_model_path()
        
        # Track initialization state and errors
        self._initialized = False
        self._init_error = None
    
    def _ensure_initialized(self) -> bool:
        """
        Ensure the model is loaded and ready for inference.
        
        Returns:
            True if model loaded successfully, False otherwise.
            
        Raises:
            RuntimeError: If model fails to load (no silent fallback).
        """
        if self._initialized:
            if self._init_error:
                raise self._init_error
            return True
        
        print(f"🎤 Loading vocal isolation model on {self.device}...")
        
        try:
            # Import required libraries
            try:
                from rotary_embedding_torch import RotaryEmbedding
                from einops import rearrange, pack, unpack, reduce, repeat
                from librosa import filters
            except ImportError as e:
                raise ImportError(
                    f"Missing required dependency for vocal isolation: {e}\n"
                    "Please install: pip install rotary-embedding-torch einops librosa"
                )
            
            # Import the model class
            from .mel_band_roformer import MelBandRoformer
            
            # Create model with ComfyUI's config
            self.model = MelBandRoformer(**MEL_BAND_ROFORMER_CONFIG).eval()
            
            # Load the checkpoint
            if self.debug:
                print(f"🔍 DEBUG: Loading checkpoint from {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Clean up state dict keys if needed (remove 'module.' prefix from DataParallel)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    key = key[7:]
                cleaned_state_dict[key] = value
            
            # Load state dict with strict=True for proper error detection
            self.model.load_state_dict(cleaned_state_dict, strict=True)
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            print(f"✅ Vocal isolation model loaded successfully")
            return True
            
        except Exception as e:
            self._initialized = True  # Mark as attempted
            self._init_error = RuntimeError(f"Failed to load vocal isolation model: {e}")
            raise self._init_error
    
    @torch.no_grad()
    def isolate(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int = 24000,
        return_instrumental: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Isolate vocals from an audio signal.
        
        Args:
            audio: Input audio as numpy array or torch tensor (mono or stereo)
            sample_rate: Sample rate of the input audio
            return_instrumental: If True, also return the instrumental track
            
        Returns:
            Isolated vocals as numpy array, or tuple of (vocals, instrumental) if return_instrumental=True
            
        Raises:
            RuntimeError: If model fails to load or inference fails.
        """
        # Ensure model is loaded (will raise if failed)
        self._ensure_initialized()
        
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Store original shape info
        original_mono = len(audio.shape) == 1
        original_length = audio.shape[-1] if original_mono else audio.shape[-1]
        original_sample_rate = sample_rate
        
        # Resample to model sample rate if needed
        if sample_rate != self.model_sample_rate:
            try:
                import librosa
                if self.debug:
                    print(f"🔍 DEBUG: Resampling from {sample_rate}Hz to {self.model_sample_rate}Hz")
                if original_mono:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.model_sample_rate)
                else:
                    audio = np.array([
                        librosa.resample(ch, orig_sr=sample_rate, target_sr=self.model_sample_rate)
                        for ch in audio
                    ])
            except Exception as e:
                raise RuntimeError(f"Failed to resample audio: {e}")
        
        # Make stereo if mono (model expects stereo)
        if original_mono:
            audio = np.stack([audio, audio], axis=0)
        elif audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)
        
        # Store for instrumental calculation
        original_audio_for_instrumental = audio.copy()
        
        # Convert to tensor: (channels, samples)
        audio_tensor = torch.from_numpy(audio).to(self.device)
        
        # Process using ComfyUI's chunked approach
        vocals = self._process_chunked(audio_tensor)
        
        # Convert back to numpy
        vocals = vocals.cpu().numpy()
        
        # Resample back to original sample rate if needed
        if original_sample_rate != self.model_sample_rate:
            try:
                import librosa
                if self.debug:
                    print(f"🔍 DEBUG: Resampling back from {self.model_sample_rate}Hz to {original_sample_rate}Hz")
                vocals = np.array([
                    librosa.resample(ch, orig_sr=self.model_sample_rate, target_sr=original_sample_rate)
                    for ch in vocals
                ])
                original_audio_for_instrumental = np.array([
                    librosa.resample(ch, orig_sr=self.model_sample_rate, target_sr=original_sample_rate)
                    for ch in original_audio_for_instrumental
                ])
            except Exception as e:
                raise RuntimeError(f"Failed to resample audio back: {e}")
        
        # Convert back to mono if input was mono
        if original_mono:
            vocals = np.mean(vocals, axis=0)
        
        # Trim or pad to match original length
        if vocals.shape[-1] > original_length:
            vocals = vocals[..., :original_length]
        elif vocals.shape[-1] < original_length:
            if original_mono:
                vocals = np.pad(vocals, (0, original_length - vocals.shape[-1]))
            else:
                vocals = np.pad(vocals, ((0, 0), (0, original_length - vocals.shape[-1])))
        
        if return_instrumental:
            if original_mono:
                original_for_sub = np.mean(original_audio_for_instrumental, axis=0)
            else:
                original_for_sub = original_audio_for_instrumental
            
            # Trim/pad original to match
            if original_for_sub.shape[-1] > original_length:
                original_for_sub = original_for_sub[..., :original_length]
            elif original_for_sub.shape[-1] < original_length:
                if original_mono:
                    original_for_sub = np.pad(original_for_sub, (0, original_length - original_for_sub.shape[-1]))
                else:
                    original_for_sub = np.pad(original_for_sub, ((0, 0), (0, original_length - original_for_sub.shape[-1])))
            
            instrumental = original_for_sub - vocals
            return vocals.astype(np.float32), instrumental.astype(np.float32)
        
        return vocals.astype(np.float32)
    
    def _process_chunked(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Process audio using ComfyUI's chunked approach for memory efficiency and quality.
        
        Args:
            audio: Input tensor of shape (channels, samples)
            
        Returns:
            Processed vocals tensor of shape (channels, samples)
        """
        channels, audio_length = audio.shape
        
        C = self.chunk_size  # 352800 (8 seconds at 44100Hz)
        N = self.num_overlap  # 2
        step = C // N  # 176400
        fade_size = C // 10  # 35280
        border = C - step  # 176400
        
        # Pad audio with reflection for border handling
        if audio_length > 2 * border and border > 0:
            audio = F.pad(audio.unsqueeze(0), (border, border), mode='reflect').squeeze(0)
        
        total_length = audio.shape[1]
        
        # Create windowing array for smooth transitions
        windowing_array = get_windowing_array(C, fade_size, self.device)
        
        # Initialize output tensors
        vocals = torch.zeros_like(audio, dtype=torch.float32)
        counter = torch.zeros_like(audio, dtype=torch.float32)
        
        # Calculate number of chunks
        num_chunks = (total_length + step - 1) // step
        
        if self.debug:
            print(f"🔍 DEBUG: Processing {num_chunks} chunks (chunk_size={C}, step={step})")
        
        # Process chunks
        for i in tqdm(range(0, total_length, step), desc="Isolating vocals", disable=not self.debug):
            # Extract chunk
            part = audio[:, i:i + C]
            length = part.shape[-1]
            
            # Pad if necessary
            if length < C:
                if length > C // 2 + 1:
                    part = F.pad(part.unsqueeze(0), (0, C - length), mode='reflect').squeeze(0)
                else:
                    part = F.pad(part.unsqueeze(0), (0, C - length), mode='constant', value=0).squeeze(0)
            
            # Run model inference: input shape (batch, channels, samples)
            x = self.model(part.unsqueeze(0))[0]  # Output: (channels, samples)
            
            # Create window for this chunk
            window = windowing_array.clone()
            if i == 0:
                window[:fade_size] = 1
            elif i + C >= total_length:
                window[-fade_size:] = 1
            
            # Accumulate with windowing
            vocals[..., i:i+length] += x[..., :length] * window[..., :length]
            counter[..., i:i+length] += window[..., :length]
        
        # Normalize by counter to handle overlapping regions
        estimated_sources = vocals / counter.clamp(min=1e-8)
        
        # Remove padding if applied
        if audio_length > 2 * border and border > 0:
            estimated_sources = estimated_sources[..., border:-border]
        
        return estimated_sources


# Convenience function for one-off vocal isolation
_global_isolator: Optional[VocalIsolator] = None


def isolate_vocals(
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = 24000,
    device: Optional[str] = None,
    return_instrumental: bool = False,
    debug: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience function to isolate vocals from audio.
    
    Uses a globally cached VocalIsolator instance for efficiency.
    
    Args:
        audio: Input audio as numpy array or torch tensor
        sample_rate: Sample rate of the input audio
        device: Device to run on (None for auto-detect)
        return_instrumental: If True, also return instrumental track
        debug: Enable debug logging
        
    Returns:
        Isolated vocals, or tuple of (vocals, instrumental) if return_instrumental=True
        
    Raises:
        RuntimeError: If model fails to load or inference fails.
    """
    global _global_isolator
    
    if _global_isolator is None:
        _global_isolator = VocalIsolator(device=device, debug=debug)
    
    return _global_isolator.isolate(audio, sample_rate, return_instrumental)


def clear_vocal_isolator_cache():
    """Clear the global VocalIsolator instance to free memory."""
    global _global_isolator
    if _global_isolator is not None:
        del _global_isolator
        _global_isolator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
