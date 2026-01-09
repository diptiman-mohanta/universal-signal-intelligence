"""
Base signal abstraction for the Universal Signal Intelligence Platform (USIP).

This module provides a simple, unified `Signal` class to represent any kind of
time-series signal (e.g., audio, ECG, EEG, radar, seismic data, etc.) in a
consistent way. The goal is to make downstream processing code agnostic to the
original domain of the signal.

Design principles:
- All signals should "look the same" to any processing pipeline.
- Domain-specific handling (e.g., loading from different file formats) belongs
  in separate loader modules, not here.
- Keep this class lightweight: minimal dependencies (only numpy + typing) and
  no heavy operations.

This makes it easy to swap signals from completely different sources without
changing the rest of the codebase.
"""

from typing import Any, Dict, Optional

import numpy as np


class Signal:
    """
    A unified representation of a time-series signal.

    The signal data is stored as a NumPy array:
    - Single-channel: shape (T,) → 1D array
    - Multi-channel: shape (C, T) → 2D array with channels as the first dimension

    Attributes
    ----------
    data : np.ndarray
        The actual signal samples.
    sr : int
        Sampling rate in Hz (accessed via property for consistency).
    channels : int
        Number of channels (inferred from data shape if not provided).
    metadata : dict
        Free-form dictionary for extra info (subject ID, device, labels, etc.).
    """

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: int,
        channels: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a new Signal instance.

        Parameters
        ----------
        data : np.ndarray
            The raw signal samples.
        sampling_rate : int
            Sampling frequency in Hertz (must be > 0).
        channels : int, optional
            Explicit number of channels. If None, it's inferred from the data shape.
        metadata : dict, optional
            Any additional information you want to attach to the signal.
        """
        # Basic type and value checks
        if not isinstance(data, np.ndarray):
            raise TypeError("Signal data must be a numpy.ndarray")
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive integer")

        self.data = data
        self.sr = sampling_rate  # using short name internally, expose via property if needed

        # Infer channels if not given
        if channels is None:
            if data.ndim == 1:
                self.channels = 1
            elif data.ndim == 2:
                self.channels = data.shape[0]
            else:
                raise ValueError("Data must be 1D (single-channel) or 2D (C, T) array")
        else:
            self.channels = channels

        # Ensure metadata is always a dict (easier to work with downstream)
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Helpful properties
    # ------------------------------------------------------------------

    @property
    def duration(self) -> float:
        """Duration of the signal in seconds."""
        num_samples = self.data.shape[-1]
        return num_samples / self.sr

    @property
    def num_samples(self) -> int:
        """Number of samples per channel."""
        return self.data.shape[-1]

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def copy(self) -> "Signal":
        """
        Return a deep copy of the signal (new data array + copied metadata).
        Useful when you want to modify a signal without affecting the original.
        """
        return Signal(
            data=self.data.copy(),
            sampling_rate=self.sr,
            channels=self.channels,
            metadata=self.metadata.copy() if self.metadata else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert key signal properties to a dictionary – great for logging or serialization."""
        return {
            "sampling_rate": self.sr,
            "channels": self.channels,
            "num_samples": self.num_samples,
            "duration_sec": self.duration,
            "metadata": self.metadata,
        }

    def summary(self) -> None:
        """Print a nice human-readable overview of the signal."""
        info = self.to_dict()
        print("Signal Summary")
        print("-" * 40)
        print(f"Sampling Rate : {info['sampling_rate']} Hz")
        print(f"Channels      : {info['channels']}")
        print(f"Samples       : {info['num_samples']:,}")
        print(f"Duration      : {info['duration_sec']:.3f} seconds")
        print(f"Metadata      : {info['metadata'] or 'None'}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Check internal consistency (shape matches declared channels).

        Raises
        ------
        ValueError
            If the data shape doesn't match the expected channel configuration.
        """
        if self.channels == 1:
            if self.data.ndim != 1:
                raise ValueError(f"Single-channel signal expects a 1D array, got shape {self.data.shape}")
        elif self.channels > 1:
            if self.data.ndim != 2 or self.data.shape[0] != self.channels:
                raise ValueError(
                    f"Multi-channel signal expects shape ({self.channels}, T), got {self.data.shape}"
                )