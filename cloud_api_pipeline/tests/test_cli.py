from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from simultrans_baseline.cli import _select_languages
from simultrans_baseline.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class _TTY:
    @staticmethod
    def isatty() -> bool:
        return True


class CLITests(unittest.TestCase):
    def test_interactive_language_selection_excludes_source_from_targets(self) -> None:
        config = load_config(ROOT / "configs" / "mock.toml")
        args = argparse.Namespace(
            source_language=None,
            target_language=None,
            no_language_prompt=False,
        )
        with (
            patch("simultrans_baseline.cli.sys.stdin", _TTY()),
            patch("builtins.input", side_effect=["2", "1"]),
            patch("builtins.print"),
        ):
            source, target = _select_languages(args, config)
        self.assertEqual(source, "Chinese")
        self.assertEqual(target, "English")


if __name__ == "__main__":
    unittest.main()
