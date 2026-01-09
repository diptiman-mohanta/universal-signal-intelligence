# universal-signal-intelligence
USIP-A unified research platform for time-series intelligence across speech, biomedical, and radar signals — combining classical DSP, deep learning, self-supervised learning, and agent-driven experimentation.

## Initial Repo Structure
```
universal-signal-intelligence/
├── data/
│   ├── raw/
│   └── processed/
│
├── signal_io/
│   ├── __init__.py
│   ├── base.py
│   ├── audio.py
│   ├── biomedical/
│   │   ├── eeg.py
│   │   ├── ecg.py
│   │   ├── emg.py
│   │   └── ppg.py
│   └── radar.py
│
├── dsp/
│   ├── __init__.py
│   ├── filters.py
│   ├── normalization.py
│   ├── spectral.py
│   └── transforms.py
│
├── preprocessing/
│   ├── pipeline.py
│   └── augmentations.py
│
├── configs/
│   └── base.yaml
│
├── evaluation/
│   └── metrics.py
│
├── README.md
└── roadmap.md
```
