from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_rag.config import load_config


class ConfigTest(unittest.TestCase):
    def test_loads_nested_scalar_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                """
llm:
  base_url: "http://localhost:1234"
  model: "unsloth:gemma-4-E4B-it-UD-MLX-4bit"
  api_key: "no_need"
  timeout: 120
  temperature: 0.0
""".strip(),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertEqual("http://localhost:1234", config["llm"]["base_url"])
        self.assertEqual("unsloth:gemma-4-E4B-it-UD-MLX-4bit", config["llm"]["model"])
        self.assertEqual("no_need", config["llm"]["api_key"])
        self.assertEqual(120, config["llm"]["timeout"])
        self.assertEqual(0.0, config["llm"]["temperature"])

    def test_missing_config_returns_empty_dict(self) -> None:
        self.assertEqual({}, load_config(Path("does-not-exist.yaml")))


if __name__ == "__main__":
    unittest.main()
