"""
Spectral-domain signal processing utilities for USIP.

This module contains classical DSP transformations that convert
time-domain signals into frequency or time-frequency representations.
These representations are essential for:
- Feature extraction
- Signal-to-image conversion
- Deep learning (CNNs, ViTs)
- Biomedical and speech analysis

Design principles:
- Works on the unified `Signal` object
- No deep learning dependencies
- NumPy-first implementation
- Preserves metadata across transformations

Author: Diptiman Mohanta
"""
import numpy as np
from signal_io.base import Signal
from typing import Optional, Tuple

def fft(signal: Signal) -> np.ndarray:
    """
    Compute the Fast Fourier Transform (FFT) of a 1D time-series signal.

    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.

    Returns
    -------
    np.ndarray
        Complex FFT values (channels, freq) or (freq,) for single-channel.
    """
    data = signal.data
    
    if signal.channels == 1:
        fft_values = np.fft.fft(data)
    else:
        fft_values = np.fft.fft(data, axis=1)
    return fft_values

def stft(
    signal: Signal,
    window_size: int = 256,
    hop_size: int = 128,
    window: str = "hann"
) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of a 1D time-series signal.

    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.
    frame_size : int
        Size of each STFT frame.
    hop_size : int
        Hop size between consecutive frames.
    window : str
        Window function to apply to each frame

    Returns
    -------
    np.ndarray
        STFT magnitude:
        - shape (freq, time) for single-channel
        - shape (channels, freq, time) for multi-channel
    """
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive integers.") 
    
    if window == "hann":
        win = np.hanning(window_size)   
    elif window == "hamming":
        win = np.hamming(window_size)       
    elif window == "blackman":
        win = np.blackman(window_size)
    elif window == "rect":  
        win = np.ones(window_size)          
    else:
        raise ValueError(f"Unsupported window type: {window}")  
    
    def _stft_1d(x: np.ndarray) -> np.ndarray:
        frames = []
        for start in range(0, len(x) - window_size + 1, hop_size):
            frame = x[start:start + window_size] * win
            spectrum = np.fft.fft(frame)
            frames.append(spectrum)
        return np.stack(frames, axis=1)  # (freq, time)
    
    data = signal.data
    if signal.channels == 1:    
        stft_values = _stft_1d(data)  # (time, freq)
    else:  
        stft_values = np.array([_stft_1d(data[ch]) for ch in range(signal.channels)], axis=0)  # (channels, time, freq)           
    return np.abs(stft_values)

def power_spectrum(signal: Signal) -> np.ndarray:
    """
    Compute the Power Spectrum of a 1D time-series signal.

    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.

    Returns
    -------
    np.ndarray
        Power spectrum values (channels, freq) or (freq,) for single-channel.
    """
    power_spectrum = fft(signal)
    power_spec = np.abs(power_spectrum) ** 2
    return power_spec

def spectrogram(
    signal: Signal,
    window_size: int = 256,
    hop_size: int = 128,
    log_scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute a magnitude spectrogram.

    This is a wrapper around STFT with optional log scaling,
    making it directly usable as an image.

    Parameters
    ----------
    signal : Signal
        Input signal.
    window_size : int
        FFT window size.
    hop_size : int
        Hop length.
    log_scale : bool
        Apply log scaling (log-magnitude).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray
        Spectrogram image.
    """
    spec = stft(signal, window_size, hop_size)
    if log_scale:
        spec = np.log(spec + eps)
    return spec

def frequency_bins(
    sampling_rate: int,
    window_size: int,
) -> np.ndarray:
    """
    Compute frequency bin centers for FFT/STFT.

    Parameters
    ----------
    sampling_rate : int
        Sampling frequency (Hz).
    window_size : int
        FFT size.

    Returns
    -------
    np.ndarray
        Frequency values in Hz.
    """
    freq_bins = np.fft.fftfreq(window_size, d=1/sampling_rate)
    return freq_bins

def band_energy(
    signal: Signal,
    band: Tuple[float, float],
    window_size: int = 256,
    hop_size: int = 128,
) -> np.ndarray:
    """
    Compute energy within a frequency band over time.

    Useful for biomedical rhythms (EEG bands) or radar Doppler bands.

    Parameters
    ----------
    signal : Signal
        Input signal.
    band : tuple (low_freq, high_freq)
        Frequency band in Hz.
    window_size : int
        STFT window size.
    hop_size : int
        Hop length.

    Returns
    -------
    np.ndarray
        Band energy over time.
    """
    spec = stft(signal, window_size, hop_size)
    freqs = frequency_bins(signal.sr, window_size)

    pos_idx = freqs >= 0
    freqs = freqs[pos_idx]
    
    if signal.channels == 1:
        spec = spec[pos_idx, :]
    else:
        spec = spec[:, pos_idx, :]

    low, high = band
    idx = np.where(freqs >= low) & (freqs <= high)[0]
    
    if signal.channels == 1:
        band_energy = np.sum(np.abs(spec[idx, :])**2, axis=0)
    else:
        band_energy = np.sum(np.abs(spec[:, idx, :])**2, axis=1)
    
    return band_energy

def mel_spectrogram(
    signal: Signal,
    window_size: int = 256,
    hop_size: int = 128,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    log_scale: bool = True,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute a Mel-scaled spectrogram.

    Parameters
    ----------
    signal : Signal
        Input signal.
    window_size : int
        FFT window size.
    hop_size : int
        Hop length.
    n_mels : int
        Number of Mel bands.
    fmin : float
        Minimum frequency (Hz).
    fmax : float or None
        Maximum frequency (Hz). If None, use Nyquist.
    log_scale : bool
        Apply log scaling (log-magnitude).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    np.ndarray
        Mel spectrogram image.
    """
    # Placeholder for actual Mel filter bank implementation
    spec = stft(signal, window_size, hop_size)
    if log_scale:
        spec = np.log(spec + eps)
    return spec  # Replace with actual Mel spectrogram computation