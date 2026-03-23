import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.server.pipeline import tonality_matcher


class TonalityMatcherTests(unittest.TestCase):
    def _write_cache(self) -> Path:
        payload = {
            "name": "test_keys",
            "source": "unit-test",
            "description": "Synthetic cache for regression coverage.",
            "embed_model": "test-model",
            "dimensions": 2,
            "content_hash": "abc123",
            "keys": [
                {
                    "key": "C major",
                    "description": "bright and direct",
                    "embedding": [1.0, 0.0],
                },
                {
                    "key": "A minor",
                    "description": "soft and reflective",
                    "embedding": [0.0, 1.0],
                },
            ],
        }
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        cache_path = Path(tmpdir.name) / "cache.json"
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return cache_path

    def test_load_cached_tonalities_reads_json_without_tonality_helper(self) -> None:
        cache_path = self._write_cache()

        cache = tonality_matcher.load_cached_tonalities(cache_path)

        self.assertEqual(cache.name, "test_keys")
        self.assertEqual(cache.embed_model, "test-model")
        self.assertEqual([entry.key for entry in cache.keys], ["C major", "A minor"])
        self.assertEqual(cache.keys[0].embedding, [1.0, 0.0])

    def test_match_prompt_to_tonalities_uses_cache_model_and_ranks_matches(self) -> None:
        cache_path = self._write_cache()

        with mock.patch.object(tonality_matcher, "embed_prompt", return_value=[0.9, 0.1]) as embed_prompt:
            result = tonality_matcher.match_prompt_to_tonalities(
                "something bright",
                cache_path=cache_path,
                top_k=2,
                embed_model=None,
            )

        embed_prompt.assert_called_once_with("something bright", embed_model="test-model")
        self.assertEqual(result.embed_model, "test-model")
        self.assertEqual([match.key for match in result.matches], ["C major", "A minor"])
        self.assertGreater(result.matches[0].score, result.matches[1].score)


if __name__ == "__main__":
    unittest.main()
