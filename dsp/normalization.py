"""
Normalization utilities for 1D time-series signals.

This module provides common normalization and scaling techniques used in
speech processing, biomedical signal analysis, and radar signal processing.

All functions operate on `Signal` objects and return new `Signal` instances
with preserved metadata.

Author: Diptiman Mohanta
"""

import numpy as np
from signal_io.base import Signal

def zscore_normalize(signal: Signal, esp: float = 1e-8) -> Signal:
    """
    Apply Z-score normalization to a 1D time-series signal.
    x_norm = (x - mean) / std
    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.

    Returns
    -------
    Signal
        Z-score normalized signal.
    """
    data = signal.data
    mean = np.mean(data)
    std = np.std(data)
    if std < esp:
        std = esp # prevent division by zero
    normalized = (data - mean) / std
    return Signal(data=normalized, 
                  sampling_rate=signal.sr,
                  signal_type=signal.signal_type,
                  metadata={ **signal.metadata, 
                            "normalization": "zscore" , 
                            "mean": float(mean), 
                            "std": float(std) 
                            },
                    )       

def minmax_normalize(
        signal: Signal,
        min_val: float = 0.0,
        max_val: float = 1.0,
        esp: float = 1e-8,
) -> Signal:
    """
    Apply min-max normalization to a signal.

    x_norm = (x - min) / (max - min)

    Parameters
    ----------
    signal : Signal
        Input signal
    min_val : float
        Lower bound of normalized range
    max_val : float
        Upper bound of normalized range
    eps : float
        Small constant to avoid division by zero

    Returns
    -------
    Signal
        Min-max normalized signal
    """
    data = signal.data 
    data_min = np.min(data)
    data_max = np.max(data)
    scaled = (data - data_min) / (data_max - data_min + esp)
    scaled = scaled * (max_val - min_val) + min_val
    return Signal(data=scaled, 
                  sampling_rate=signal.sr,
                  signal_type=signal.signal_type,
                  metadata={ **signal.metadata, 
                            "normalization": "minmax" , 
                            "data_min": float(data_min), 
                            "data_max": float(data_max),
                            "range": (min_val, max_val),
                            },
                    ) 

def mean_normalize(signal: Signal, esp: float = 1e-8) -> Signal:
    """
    Apply mean normalization to a 1D time-series signal.
    x_norm = (x - mean) / (max - min)
    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.

    Returns
    -------
    Signal
        Mean normalized signal.
    """
    data = signal.data
    mean = np.mean(data)
    max_abs = np.max(np.abs(data))
    normalized = (data - mean) / (max_abs + esp)
    return Signal(data=normalized, 
                  sampling_rate=signal.sr,
                  signal_type=signal.signal_type,
                  metadata={ **signal.metadata, 
                            "normalization": "mean" , 
                            "mean": float(mean), 
                            "max_abs": float(max_abs)    
                            },
                    )

def unit_energy_normalize(signal: Signal, esp: float = 1e-8) -> Signal:
    """
    Apply unit energy normalization to a 1D time-series signal.
    x_norm = x / ||x||
    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.

    Returns
    -------
    Signal
        Unit energy normalized signal.
    """
    data = signal.data
    energy = np.sqrt(np.sum(data ** 2))
    normalized = data / (energy + esp)
    return Signal(data=normalized, 
                  sampling_rate=signal.sr,
                  signal_type=signal.signal_type,
                  metadata={ **signal.metadata, 
                            "normalization": "unit_energy" , 
                            "energy": float(energy)    
                            },
                    )

def peak_normalize(signal: Signal, esp: float = 1e-8) -> Signal:
    """
    Apply peak normalization to a 1D time-series signal.
    x_norm = x / max(|x|)
    Parameters
    ----------
    signal : Signal
        Input 1D time-series signal.

    Returns
    -------
    Signal
        Peak normalized signal.
    """
    data = signal.data 
    peak = np.max(np.abs(data))
    normalized = data / (peak + esp)
    return Signal(data=normalized, 
                  sampling_rate=signal.sr,
                  signal_type=signal.signal_type,
                  metadata={ **signal.metadata, 
                            "normalization": "peak" , 
                            "peak": float(peak)    
                            },
                    )
