"""Fail-closed authored-text and actual CSS semantic-contrast gates."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SKIP = {
    ".git",
    ".venv",
    ".venv312",
    "node_modules",
    ".next",
    ".next-dev",
    "out",
    ".cache",
    ".runtime",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".hypothesis",
    ".openai",
    "data",
    "results",
    "claude",
    "artifacts",
}
TEXT = {".py", ".md", ".html", ".j2", ".ts", ".tsx", ".css", ".yml", ".yaml", ".toml"}
WORDS = (
    "lever" + "age",
    "del" + "ve",
    "seam" + "less",
    "cutting" + " edge",
    "comprehen" + "sive",
    "state" + " of the art",
    "empo" + "wer",
    "unlo" + "ck",
    "ele" + "vate",
    "stream" + "line",
    "holi" + "stic",
    "syner" + "gy",
    "game" + " changing",
    "next" + " generation",
    "best" + " in class",
    "util" + "ize",
    "facil" + "itate",
    "in" + " today's",
    "it is" + " worth noting",
)
VOCABULARY = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in WORDS) + r")(?:s|d|ing)?\b", re.IGNORECASE
)


def authored_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name not in SKIP]
        for name in names:
            path = Path(directory) / name
            if path.suffix in TEXT and "public" not in path.relative_to(root).parts:
                files.append(path)
    if not files or not any(path.read_text(encoding="utf-8-sig").strip() for path in files):
        raise ValueError("No nonempty authored text to validate")
    return sorted(files)


def punctuation_errors(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{line}"
        for path in authored_files(root)
        for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if chr(0x2014) in text
    ]


def vocabulary_errors(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{line}"
        for path in authored_files(root)
        for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1)
        if VOCABULARY.search(text)
    ]


def luminance(hex_color: str) -> float:
    components = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    channels = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in components
    ]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast(first: str, second: str) -> float:
    a, b = luminance(first), luminance(second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def palette_checks(css: str) -> list[dict[str, Any]]:
    patterns = {"light": r":root\s*\{([^}]+)\}", "dark": r':root\[data-theme="dark"\]\s*\{([^}]+)\}'}
    checks: list[dict[str, Any]] = []
    for theme, pattern in patterns.items():
        match = re.search(pattern, css)
        if match is None:
            raise ValueError(f"Missing nonempty {theme} palette")
        colors = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})(?:;|\s)", match.group(1)))
        for foreground in ("ink", "muted", "burgundy", "strong"):
            for background in ("paper", "panel"):
                if foreground not in colors or background not in colors:
                    raise ValueError(f"Missing palette token: {theme} {foreground}/{background}")
                ratio = contrast(colors[foreground], colors[background])
                checks.append(
                    {
                        "theme": theme,
                        "foreground": foreground,
                        "background": background,
                        "contrast": ratio,
                        "threshold": 4.5,
                        "passed": ratio >= 4.5,
                    }
                )
    return checks


def enforce(kind: str, root: Path = ROOT) -> None:
    if kind == "palette":
        css = (root / "web" / "app" / "globals.css").read_text(encoding="utf-8")
        checks = palette_checks(css)
        errors = [str(check) for check in checks if not check["passed"]]
    elif kind == "punctuation":
        errors = punctuation_errors(root)
    elif kind == "vocabulary":
        errors = vocabulary_errors(root)
    else:
        raise ValueError(f"Unknown gate: {kind}")
    if errors:
        raise ValueError(f"{kind} gate failed:\n" + "\n".join(errors))
    print(f"{kind} gate passed")
