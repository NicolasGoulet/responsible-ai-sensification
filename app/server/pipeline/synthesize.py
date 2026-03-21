"""synthesize.py: Read a GenerationAnalysis JSON and produce a WAV file, or play live.

Usage:
    # Batch mode (writes WAV):
    python synthesize.py <input.json> [--output-dir audio]

    # Live mode (reads MusicalEvent NDJSON from stdin):
    ... | python synthesize.py --live [--mode timed|sustain]
"""
import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from app.server.pipeline.audio_utils import (
    SAMPLE_RATE,
    SAMPLES_PER_TOKEN,
    feature_to_frequency,
    generate_token_audio,
)


def synthesize_additive(input_path: Path, output_dir: Path) -> Path:
    with open(input_path) as f:
        data = json.load(f)

    tokens = data["generated_tokens"]
    total_samples = len(tokens) * SAMPLES_PER_TOKEN
    audio = np.zeros(total_samples, dtype=np.float64)

    for i, tok in enumerate(tokens):
        start = i * SAMPLES_PER_TOKEN
        end = start + SAMPLES_PER_TOKEN
        notes = [
            {"freq": feature_to_frequency(f["index"]), "amplitude": f["activation"]}
            for f in tok["active_features"]
        ]
        segment = generate_token_audio(notes)
        audio[start:end] = segment

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    audio_int16 = (audio * 32767).astype(np.int16)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + ".wav")
    sf.write(str(output_path), audio_int16, SAMPLE_RATE, subtype="PCM_16")
    print(f"Written: {output_path}")
    return output_path


def _ndjson_tokens(source):
    """Yield token event dicts from an NDJSON source (file or stdin lines)."""
    for line in source:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if data.get("type") == "token":
            yield data


def live_timed(source):
    """Play each token's audio for exactly TOKEN_DURATION seconds."""
    import sounddevice as sd

    for event in _ndjson_tokens(source):
        notes = event.get("notes", [])
        segment = generate_token_audio(notes)
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak
        sd.play(segment.astype(np.float32), samplerate=SAMPLE_RATE, blocking=True)


def live_sustain(source):
    """Play audio continuously, hard-cutting to the new sound on each token arrival."""
    import sounddevice as sd

    silence = np.zeros(SAMPLES_PER_TOKEN, dtype=np.float32)
    lock = threading.Lock()
    state = {"buf": silence, "pos": 0}

    def callback(outdata, frames, time_info, status):
        with lock:
            buf = state["buf"]
            pos = state["pos"]
            remaining = len(buf) - pos
            if frames <= remaining:
                out = buf[pos : pos + frames]
                state["pos"] = pos + frames
            else:
                overflow = frames - remaining
                out = np.concatenate([buf[pos:], buf[:overflow]])
                state["pos"] = overflow
        outdata[:, 0] = out

    try:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            latency="high",
            callback=callback,
        )
    except sd.PortAudioError as e:
        print(
            f"Audio device error: {e}\n"
            "On WSL2, install PulseAudio: sudo apt install pulseaudio\n"
            "Then start it: pulseaudio --start",
            file=sys.stderr,
        )
        sys.exit(1)

    with stream:
        try:
            for event in _ndjson_tokens(source):
                notes = event.get("notes", [])
                segment = generate_token_audio(notes)
                peak = np.max(np.abs(segment))
                if peak > 0:
                    segment = segment / peak
                with lock:
                    state["buf"] = segment.astype(np.float32)
                    state["pos"] = 0
        except KeyboardInterrupt:
            pass


METHODS = {
    "additive": synthesize_additive,
}


def main():
    parser = argparse.ArgumentParser(description="Synthesize SAE feature audio from JSON or stdin")
    parser.add_argument(
        "input", nargs="?", type=Path, help="Input JSON file (omit with --live to read stdin)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("audio"), help="Output directory (batch mode)"
    )
    parser.add_argument(
        "--method",
        default="additive",
        choices=list(METHODS.keys()),
        help="Synthesis method (default: additive)",
    )
    parser.add_argument("--live", action="store_true", help="Play audio live from NDJSON stdin")
    parser.add_argument(
        "--mode",
        default="timed",
        choices=["timed", "sustain"],
        help="Live playback mode: timed (fixed 0.5s/token) or sustain (hold until next token)",
    )
    args = parser.parse_args()

    if args.live:
        source = sys.stdin if args.input is None else open(args.input)
        if args.mode == "sustain":
            live_sustain(source)
        else:
            live_timed(source)
    else:
        if args.input is None:
            parser.error("input is required in batch mode (omit only with --live)")
        METHODS[args.method](args.input, args.output_dir)


if __name__ == "__main__":
    main()
