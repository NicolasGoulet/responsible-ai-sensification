"""synthesize.py: Read a GenerationAnalysis JSON and produce a WAV file.

Usage:
    python synthesize.py <input.json> [--output-dir audio]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

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
    # Clamp instrument in case of edge cases
    instrument = min(instrument, NUM_INSTRUMENTS - 1)
    # Each instrument has its own log-mapped sub-range
    # We use the same formula across all instruments based on local index
    freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN) ** (local_index / FEATURES_PER_INSTRUMENT)
    return freq


def generate_token_audio(active_features: list[dict]) -> np.ndarray:
    """Sum sine waves for all active features, return normalized float32 array."""
    t = np.linspace(0, TOKEN_DURATION, SAMPLES_PER_TOKEN, endpoint=False)
    buffer = np.zeros(SAMPLES_PER_TOKEN, dtype=np.float64)

    for feat in active_features:
        freq = feature_to_frequency(feat["index"])
        amplitude = feat["activation"]
        buffer += amplitude * np.sin(2 * np.pi * freq * t)

    return buffer


def synthesize_additive(input_path: Path, output_dir: Path) -> Path:
    with open(input_path) as f:
        data = json.load(f)

    tokens = data["generated_tokens"]
    total_samples = len(tokens) * SAMPLES_PER_TOKEN
    audio = np.zeros(total_samples, dtype=np.float64)

    for i, tok in enumerate(tokens):
        start = i * SAMPLES_PER_TOKEN
        end = start + SAMPLES_PER_TOKEN
        segment = generate_token_audio(tok["active_features"])
        audio[start:end] = segment

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    audio_int16 = (audio * 32767).astype(np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + ".wav")
    sf.write(str(output_path), audio_int16, SAMPLE_RATE, subtype="PCM_16")
    print(f"Written: {output_path}")
    return output_path


METHODS = {
    "additive": synthesize_additive,
}


def main():
    parser = argparse.ArgumentParser(description="Synthesize SAE feature audio from JSON")
    parser.add_argument("input", type=Path, help="Input JSON file")
    parser.add_argument("--output-dir", type=Path, default=Path("audio"), help="Output directory")
    parser.add_argument(
        "--method",
        default="additive",
        choices=list(METHODS.keys()),
        help="Synthesis method (default: additive)",
    )
    args = parser.parse_args()
    METHODS[args.method](args.input, args.output_dir)


if __name__ == "__main__":
    main()
