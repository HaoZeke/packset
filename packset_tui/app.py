"""Packset pane: Textual DataTable over packsetd HTTP.

Not WorkGraph. Not todos. Not the vault.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header

DEFAULT_URL = "http://127.0.0.1:8761"


def packset_url() -> str:
    return os.environ.get("PACKSET_URL") or os.environ.get("INSIDE_MEMORY_URL") or DEFAULT_URL


def workspace_name() -> str:
    return os.environ.get("PACKSET_WORKSPACE") or "global"


def fetch_atoms(base: str, workspace: str) -> tuple[list[dict], str]:
    q = urllib.parse.urlencode({"workspace": workspace})
    url = f"{base.rstrip('/')}/v1/atoms?{q}"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return [], str(exc)
    atoms = body.get("atoms") if isinstance(body, dict) else None
    if not isinstance(atoms, list):
        return [], "bad atoms payload"
    return [a for a in atoms if isinstance(a, dict)], ""


def bump_ts(base: str, workspace: str, atom_id: str) -> str:
    url = f"{base.rstrip('/')}/v1/atoms/update"
    payload = json.dumps(
        {"workspace": workspace, "id": atom_id, "fields": {}}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return str(exc)
    return ""


class PacksetApp(App[None]):
    CSS_PATH = Path(__file__).with_name("storm.tcss")
    TITLE = "packset"
    BINDINGS = [
        Binding("b", "bump", "Bump ts"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.base = packset_url()
        self.workspace = workspace_name()
        self._ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable()
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("id", "kind", "text", "ts", "urgency")
        self.refresh_table()

    def refresh_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        atoms, err = fetch_atoms(self.base, self.workspace)
        self._ids = []
        if err:
            table.add_row("—", "offline", err, "", "")
            return
        for atom in atoms:
            aid = str(atom.get("id") or "")
            self._ids.append(aid)
            urgency = atom.get("urgency")
            table.add_row(
                aid[:12],
                str(atom.get("kind") or ""),
                str(atom.get("text") or "").replace("\n", " ")[:48],
                str(atom.get("ts") or ""),
                "" if urgency is None else str(urgency),
            )

    def action_refresh(self) -> None:
        self.refresh_table()

    def action_bump(self) -> None:
        table = self.query_one(DataTable)
        if not self._ids:
            self.notify("no atom")
            return
        coord = table.cursor_coordinate
        if coord.row < 0 or coord.row >= len(self._ids):
            self.notify("select a row")
            return
        err = bump_ts(self.base, self.workspace, self._ids[coord.row])
        self.notify(err or "ts bumped")
        self.refresh_table()


def main() -> None:
    PacksetApp().run()


if __name__ == "__main__":
    main()
