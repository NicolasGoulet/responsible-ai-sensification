"""pitch_policy.py: Minimal tonal pitch policy for token-level note events.

This module is intentionally small and easy to modify.

It separates two kinds of inputs:
1. Prompt-level tonal context (chosen tonalities from the tonality matcher)
2. Token / feature-level event information (feature index, activation, etc.)

The first implementation here is deliberately conservative:
- derive a provisional MIDI note from the feature index
- use the strongest prompt-level tonality as the tonal center
- bias the provisional note toward the nearest in-scale note
- keep a little off-scale motion when the bias is weak

This gives the project a simple, inspectable place to evolve pitch logic without
burying tonal behavior inside transform.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from app.server.pipeline.tonality_matcher import TonalityMatch

# Basic pitch-class maps. This is intentionally small and explicit so it is easy to change.
PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


@dataclass(frozen=True)
class TonalityContext:
    """Prompt-level tonal information.

    `matches` should usually come from the prompt tonality matcher. We keep the whole list
    so future policies can use top-k tonalities, not just the winner.
    """

    matches: list[TonalityMatch]

    @property
    def primary(self) -> TonalityMatch:
        if not self.matches:
            raise ValueError("TonalityContext requires at least one tonal match")
        return self.matches[0]


@dataclass(frozen=True)
class TokenPitchInput:
    """Small token / feature-level input bundle for pitch selection."""

    feature_index: int
    activation: float
    token_id: int = 0
    token: str = ""
    cluster: int | None = None
    instrument: str | None = None
    l0: int = 0
    elapsed_ms: int = 0


@dataclass(frozen=True)
class PitchDecision:
    """The result of the pitch policy.

    `chosen_midi` is the tonal-biased pitch the rest of the pipeline can render.
    `raw_midi` keeps the pre-bias value to make experimentation easier.
    """

    chosen_midi: int
    raw_midi: int
    key_name: str
    mode: str
    pitch_class: int
    used_scale_bias: bool


def _parse_key_name(key_name: str) -> tuple[int, str]:
    """Return (root_pitch_class, mode) from names like 'C major' or 'Bb minor'."""
    parts = key_name.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Could not parse tonal key name: {key_name!r}")

    tonic = parts[0]
    mode = parts[1].lower()
    if tonic not in PITCH_CLASS:
        raise ValueError(f"Unsupported tonic spelling in key name: {tonic!r}")
    if mode not in {"major", "minor"}:
        raise ValueError(f"Unsupported mode in key name: {mode!r}")
    return PITCH_CLASS[tonic], mode


def _scale_pitch_classes(root_pc: int, mode: str) -> list[int]:
    intervals = MAJOR_INTERVALS if mode == "major" else MINOR_INTERVALS
    return [int((root_pc + interval) % 12) for interval in intervals]


def _feature_index_to_midi(feature_index: int, midi_low: int = 48, midi_high: int = 84) -> int:
    """Map a feature index to a provisional MIDI note.

    This is intentionally simple and stable:
    - feature identity determines a repeatable pitch lane
    - the range is bounded to a musical register that is easy to hear
    """
    width = max(1, midi_high - midi_low)
    return midi_low + (feature_index % (width + 1))


def _nearest_scale_midi(raw_midi: int, allowed_pitch_classes: list[int]) -> int:
    """Return the nearest MIDI note whose pitch class belongs to the target scale."""
    best = raw_midi
    best_distance = None
    for candidate in range(raw_midi - 12, raw_midi + 13):
        if candidate % 12 not in allowed_pitch_classes:
            continue
        distance = abs(candidate - raw_midi)
        if best_distance is None or distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def choose_pitch(
    tonal_context: TonalityContext,
    token_input: TokenPitchInput,
    *,
    strong_bias_threshold: float = 0.35,
) -> PitchDecision:
    """Choose a tonal-biased MIDI pitch for one token / feature event.

    Minimal first policy:
    1. pick the strongest prompt-level tonality
    2. map feature index to a provisional MIDI note
    3. if the tonality match is reasonably strong, snap to the nearest scale tone
    4. if the tonality match is weak, keep the raw note to allow tonal ambiguity

    This gives the project a "soft bias" instead of a total hard lock.
    """
    primary = tonal_context.primary
    root_pc, mode = _parse_key_name(primary.key)
    allowed_pitch_classes = _scale_pitch_classes(root_pc, mode)

    raw_midi = _feature_index_to_midi(token_input.feature_index)
    use_bias = primary.score >= strong_bias_threshold
    chosen_midi = _nearest_scale_midi(raw_midi, allowed_pitch_classes) if use_bias else raw_midi

    return PitchDecision(
        chosen_midi=chosen_midi,
        raw_midi=raw_midi,
        key_name=primary.key,
        mode=mode,
        pitch_class=int(chosen_midi % 12),
        used_scale_bias=use_bias,
    )
