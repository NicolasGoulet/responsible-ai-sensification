"""audio_utils.py: Shared audio synthesis utilities."""
import numpy as np

SAMPLE_RATE = 22050
TOKEN_DURATION = 0.5  # seconds per token
SAMPLES_PER_TOKEN = int(SAMPLE_RATE * TOKEN_DURATION)

NUM_INSTRUMENTS = 5
FEATURES_PER_INSTRUMENT = 13000  # each instrument covers 13k features
FREQ_MIN = 20.0
FREQ_MAX = 20000.0


def feature_to_frequency(feature_index: int) -> float:
    """Map a feature index (0-64999) to a frequency in Hz using log scale."""
    instrument = feature_index // FEATURES_PER_INSTRUMENT
    local_index = feature_index % FEATURES_PER_INSTRUMENT
    instrument = min(instrument, NUM_INSTRUMENTS - 1)
    freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN) ** (local_index / FEATURES_PER_INSTRUMENT)
    return freq


def _synthesize_note(t: np.ndarray, freq: float, amplitude: float, instrument: str | None) -> np.ndarray:
    """Synthesize a single note with instrument-specific timbre."""
    if instrument == "piano":
        wave = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    elif instrument == "guitar":
        wave = (
            np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.2 * np.sin(2 * np.pi * 3 * freq * t)
        )
    elif instrument == "bass":
        wave = np.sin(2 * np.pi * freq * t) + 0.6 * np.sin(2 * np.pi * (freq / 2) * t)
    elif instrument == "strings":
        wave = np.sin(2 * np.pi * freq * t) + np.sin(2 * np.pi * (freq + 2.0) * t)
    elif instrument == "pad":
        envelope = np.sin(np.pi * t / TOKEN_DURATION)
        wave = np.sin(2 * np.pi * freq * t) * envelope
    elif instrument == "bell":
        decay = np.exp(-3 * t / TOKEN_DURATION)
        wave = (
            np.sin(2 * np.pi * freq * t)
            + 0.4 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.2 * np.sin(2 * np.pi * 5 * freq * t)
        ) * decay
    elif instrument == "flute":
        wave = np.sin(2 * np.pi * freq * t)
    elif instrument == "brass":
        wave = (
            np.sin(2 * np.pi * freq * t)
            + 0.7 * np.sin(2 * np.pi * 2 * freq * t)
            + 0.5 * np.sin(2 * np.pi * 3 * freq * t)
            + 0.3 * np.sin(2 * np.pi * 4 * freq * t)
        )
    else:
        wave = np.sin(2 * np.pi * freq * t)

    return amplitude * wave


def generate_token_audio(notes: list[dict]) -> np.ndarray:
    """Sum waveforms for all notes, return float64 array.

    Each note is a dict with keys: freq, amplitude, and optionally instrument.
    """
    t = np.linspace(0, TOKEN_DURATION, SAMPLES_PER_TOKEN, endpoint=False)
    buffer = np.zeros(SAMPLES_PER_TOKEN, dtype=np.float64)

    for note in notes:
        freq = note["freq"]
        amplitude = note["amplitude"]
        instrument = note.get("instrument")
        buffer += _synthesize_note(t, freq, amplitude, instrument)

    return buffer
