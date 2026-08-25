"""Seat CSS: GrokDay on prefer-light / GROK_THEME=auto, else Tokyo Night Storm."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

TUI_DIR = Path(__file__).resolve().parent

_LIGHT = frozenset({"grokday", "light", "day"})
_DARK = frozenset(
    {"tokyonight", "tokyo-night", "groknight", "dark", "oscura-midnight", "storm"}
)


def seat_prefers_light() -> bool:
    name = os.environ.get("GROK_THEME", "").strip().lower()
    if name in _LIGHT:
        return True
    if name in _DARK:
        return False
    scheme = os.environ.get("GROKOS_COLOR_SCHEME", "").strip().lower()
    if scheme in {"light", "prefer-light"}:
        return True
    if scheme in {"dark", "prefer-dark"}:
        return False
    try:
        out = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        text = out.stdout or ""
        if "prefer-light" in text:
            return True
        if "prefer-dark" in text:
            return False
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def css_files() -> list[Path]:
    if seat_prefers_light():
        return [TUI_DIR / "grokday.tcss"]
    return [TUI_DIR / "tokyo_night_storm.tcss", TUI_DIR / "storm.tcss"]


def apply_seat_theme(app: object) -> None:
    """Textual 8 defaults to textual-dark; that paints DataTable/Footer Storm."""
    app.theme = "textual-light" if seat_prefers_light() else "tokyo-night"  # type: ignore[attr-defined]


CSS_FILES = css_files()
