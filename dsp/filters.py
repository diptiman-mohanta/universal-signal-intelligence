"""
Filtering utilities for signals.

All filters operate on Signal objects and return new Signal instances with preserved metadata.

Author: Diptiman Mohanta
"""

import numpy as np
import scipy.signal as sg
from scipy.signal import butter, filtfilt, iirnotch
from scipy.ndimage import median_filter
import warnings

from signal_io.base import Signal

def butterworth_filter(
    signal: Signal,
    filter_type: str = "bandpass",
    low_freq: float | None = None,
    high_freq: float | None = None,
    order: int = 4,
) -> Signal:
    """
    Apply a Butterworth filter to the input Signal.

    Parameters
    ----------
    signal : Signal
        Input Signal object to be filtered.
    filter_type : str, default="bandpass"
        Type of Butterworth filter: "lowpass", "highpass", "bandpass", or "bandstop".       
    low_freq : float, optional
        Low cutoff frequency in Hz (required for "highpass", "bandpass", "bandstop").                       
    high_freq : float, optional
        High cutoff frequency in Hz (required for "lowpass", "bandpass", "bandstop").   
    order : int, default=4
        Order of the Butterworth filter.
    Returns 
    -------
    Signal
        New Signal object containing the filtered data.
    """ 
    data = signal.data
    fs = signal.fs
    nyquist = fs / 2.0

    if filter_type == "lowpass":
        if high_freq is None:
            raise ValueError("high_freq must be specified for lowpass filter")
        high_freq = min(high_freq, nyquist * 0.99)
        b, a = butter(order, high_freq / nyquist, btype='low')
    elif filter_type == "highpass":
        if low_freq is None:
            raise ValueError("low_freq must be specified for highpass filter")
        if low_freq >= nyquist:
            raise ValueError("low_freq must be less than Nyquist frequency")
        b, a = butter(order, low_freq / nyquist, btype='high')
    elif filter_type == "bandpass":
        if low_freq is None or high_freq is None:
            raise ValueError("Both low_freq and high_freq must be specified for bandpass filter")
        if low_freq >= high_freq:
            raise ValueError("low_freq must be less than high_freq")
        high_freq = min(high_freq, nyquist * 0.99)
        b, a = butter(order, [low_freq / nyquist, high_freq / nyquist], btype='band')       
    elif filter_type == "bandstop":
        if low_freq is None or high_freq is None:
            raise ValueError("Both low_freq and high_freq must be specified for bandstop filter")
        elif low_freq >= high_freq:
            raise ValueError("low_freq must be less than high_freq")
        high_freq = min(high_freq, nyquist * 0.99)
        b, a = butter(order, [low_freq / nyquist, high_freq / nyquist], btype='bandstop')

    else:
        raise ValueError("Invalid filter_type. Must be 'lowpass', 'highpass', 'bandpass', or 'bandstop'.")  
    
    filtered = filtfilt(b, a, data)

    return Signal(
        data=filtered,
        fs=fs,
        signal_type=signal.signal_type,
        metadata={**signal.metadata, "filter": f"butterworth_{filter_type}"},
    )

def notch_filter(
        signal: Signal,
        notch_freq: float = 50.0,
        quality_factor: float = 30.0,
    ) -> Signal:
    """
    Apply a notch filter to the input Signal to remove a specific frequency.
    Parameters
    ----------
    signal : Signal
        Input Signal object to be filtered.
    notch_freq : float, default=50.0
        Frequency to be removed from the signal in Hz.
    quality_factor : float, default=30.0
        Quality factor of the notch filter (higher means narrower notch).
    Returns
    -------
    Signal
        New Signal object containing the filtered data.
    """
    data = signal.data
    fs = signal.fs
    nyquist = fs / 2.0

    if notch_freq >= nyquist:
        raise ValueError("notch_freq must be less than Nyquist frequency")  
        return signal

    # Design the notch filter
    b, a = iirnotch(notch_freq, quality_factor, fs)

    # Apply the filter
    filtered = filtfilt(b, a, data)

    return Signal(
        data=filtered,
        fs=fs,
        signal_type=signal.signal_type,
        metadata={**signal.metadata, "filter": f"notch_{notch_freq}Hz"},
    )