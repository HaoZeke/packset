"""Packset pane: Textual DataTable over packsetd HTTP.

Not WorkGraph. Not todos. Not the vault.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Footer, Header, Static

from packset_tui.atoms import (
    BASE_COLUMNS,
    bump_ts,
    fetch_atoms,
    packset_url,
    table_columns,
    table_row,
    workspace_name,
)
from packset_tui.theme import CSS_FILES


class PacksetApp(App[None]):
    CSS_PATH = CSS_FILES
    TITLE = "packset"
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("b", "bump", "Bump ts"),
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        base: str | None = None,
        workspace: str | None = None,
    ) -> None:
        super().__init__()
        self.base = (base or packset_url()).rstrip("/")
        self.workspace = workspace or workspace_name(self.base)
        self._ids: list[str] = []
        self._columns: tuple[str, ...] = BASE_COLUMNS

    def compose(self) -> ComposeResult:
        self.sub_title = f"{self.base}  {self.workspace}"
        yield Header()
        yield DataTable()
        yield Static(id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.refresh_table()

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def refresh_table(self) -> None:
        table = self.query_one(DataTable)
        atoms, err = fetch_atoms(self.base, self.workspace)
        columns = table_columns(atoms)
        if columns != self._columns or not table.columns:
            table.clear(columns=True)
            table.add_columns(*columns)
            self._columns = columns
        else:
            table.clear()
        self._ids = []
        if err:
            self._set_status(f"offline  {err}")
            return
        for atom in atoms:
            aid = str(atom.get("id") or "")
            self._ids.append(aid)
            table.add_row(*table_row(atom, columns), key=aid)
        self._set_status(f"{len(atoms)} atoms  {self.workspace}")

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
        updated, err = bump_ts(self.base, self.workspace, self._ids[coord.row])
        self.notify(err or "ts bumped")
        if updated is not None:
            self.refresh_table()
