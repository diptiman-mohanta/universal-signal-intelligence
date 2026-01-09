"""
Audio signal loading utilities for the Universal Signal Intelligence Platform (USIP).

This module is responsible for loading common audio file formats (primarily WAV and FLAC)
and converting them into the unified `Signal` object defined in `signal_io.base`.

Key responsibilities:
- Load audio files using librosa
- Optional mono conversion (default: enabled)
- Optional resampling to a target sampling rate
- Attach useful metadata (source path, domain tag)
- Always return a clean `Signal` instance

Important design rules:
- All audio-specific I/O logic stays here — keeps the core `Signal` class clean.
- No heavy DSP (filtering, spectrograms, feature extraction) in this module.
- Utility functions like `ensure_mono` and `resample` are provided for convenience
  but can be used independently.

This separation allows the rest of the platform to treat audio just like any other
signal type (ECG, EEG, radar, etc.).
"""

from typing import Optional, Dict, Any

import numpy as np
import librosa

# Import the unified Signal class from the base module
from signal_io.base import Signal


def load_audio(
    filepath: str,
    target_sr: Optional[int] = None,
    mono: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Signal:
    """
    Load an audio file and return it as a standardized `Signal` object.

    Uses librosa under the hood, which supports many formats (WAV, FLAC, MP3, etc.)
    and automatically normalizes samples to float32 in the range [-1.0, 1.0].

    Parameters
    ----------
    filepath : str
        Path to the audio file on disk.
    target_sr : int, optional
        Desired sampling rate in Hz. If None, keeps the file's original rate.
        If provided, librosa will resample on load (efficient!).
    mono : bool, default=True
        If True, converts multi-channel audio to mono by averaging channels.
        Set to False if you need to preserve stereo/multi-channel layout.
    metadata : dict, optional
        Extra metadata you want to attach (e.g., {"genre": "speech", "speaker": "001"}).

    Returns
    -------
    Signal
        A fully populated `Signal` instance ready for downstream processing.
    """
    # Load audio with librosa — returns waveform and sampling rate
    waveform, sr = librosa.load(
        filepath,
        sr=target_sr,    # None → native rate; int → resample on load
        mono=mono,       # handles channel averaging if True
    )

    # waveform shape handling:
    # - After mono=True  → (T,)
    # - After mono=False → (C, T) for multi-channel
    if waveform.ndim == 1:
        channels = 1
        data = waveform
    else:
        channels = waveform.shape[0]
        data = waveform

    # Build metadata dictionary — always include source and domain for traceability
    signal_metadata = metadata or {}
    signal_metadata.update(
        {
            "source_file": filepath,
            "domain": "audio",
            "original_sr": sr if target_sr is None else "resampled",
        }
    )

    return Signal(
        data=data,
        sampling_rate=sr,
        channels=channels,
        metadata=signal_metadata,
    )


def ensure_mono(signal: Signal) -> Signal:
    """
    Force a signal to mono by averaging across channels.

    Useful when downstream processing expects single-channel input
    (e.g., many speech models or simple visualizations).

    Parameters
    ----------
    signal : Signal
        The input signal (can be mono or multi-channel).

    Returns
    -------
    Signal
        A new mono `Signal` object. The original signal is unchanged.
    """
    # If already mono, just return a copy to keep interface consistent
    if signal.channels == 1:
        return signal.copy()

    # Average across the channel dimension: (C, T) → (T,)
    mono_data = np.mean(signal.data, axis=0, dtype=signal.data.dtype)

    # Preserve all metadata, but note that we converted to mono
    new_metadata = signal.metadata.copy()
    new_metadata["conversion"] = new_metadata.get("conversion", []) + ["to_mono_by_averaging"]

    return Signal(
        data=mono_data,
        sampling_rate=signal.sr,
        channels=1,
        metadata=new_metadata,
    )


def resample(signal: Signal, target_sr: int) -> Signal:
    """
    Resample a signal to a new target sampling rate using librosa.

    Works for both mono and multi-channel signals by resampling each channel
    independently (preserves phase alignment).

    Parameters
    ----------
    signal : Signal
        Input signal to resample.
    target_sr : int
        Desired sampling rate in Hz (must be positive integer).

    Returns
    -------
    Signal
        A new resampled `Signal` object. Original remains untouched.
    """
    if target_sr <= 0:
        raise ValueError("Target sampling rate must be a positive integer")

    # No work needed if rates match
    if signal.sr == target_sr:
        return signal.copy()

    if signal.channels == 1:
        # Simple case: 1D array
        resampled_data = librosa.resample(
            signal.data,
            orig_sr=signal.sr,
            target_sr=target_sr,
        )
        resampled_channels = 1
    else:
        # Resample each channel separately and stack back to (C, T_new)
        resampled_channels = []
        for channel in signal.data:  # iterate over channels
            resampled_channel = librosa.resample(
                channel,
                orig_sr=signal.sr,
                target_sr=target_sr,
            )
            resampled_channels.append(resampled_channel)

        resampled_data = np.vstack(resampled_channels)
        resampled_channels = signal.channels

    # Update metadata to track the operation
    new_metadata = signal.metadata.copy()
    new_metadata["conversion"] = new_metadata.get("conversion", []) + [
        f"resampled_{signal.sr}_to_{target_sr}Hz"
    ]

    return Signal(
        data=resampled_data,
        sampling_rate=target_sr,
        channels=resampled_channels,
        metadata=new_metadata,
    )