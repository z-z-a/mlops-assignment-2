"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


def _example_values(conn: sqlite3.Connection, table: str, col: str,
                    limit: int = 3, maxlen: int = 40) -> str:
    """Return a `-- e.g. ...` comment with up to `limit` distinct sample values.

    Showing real stored values lets the model match string literals exactly
    (casing/spelling/coding) and pick the right column - the most common
    silent-wrong cause on BIRD. Returns "" if there's nothing useful to show
    (blob/empty) or the probe fails (never breaks schema rendering).
    """
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(col)} FROM {_q(table)} "
            f"WHERE {_q(col)} IS NOT NULL LIMIT {limit}"
        ).fetchall()
    except sqlite3.Error:
        return ""
    vals: list[str] = []
    for (v,) in rows:
        if isinstance(v, bytes):
            return ""  # blob column - examples aren't useful
        if isinstance(v, str):
            s = v.replace("\n", " ").strip()
            if len(s) > maxlen:
                s = s[:maxlen] + "..."
            vals.append("'" + s.replace("'", "''") + "'")
        else:
            vals.append(str(v))
    return "-- e.g. " + ", ".join(vals) if vals else ""


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?")

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            # Each entry is (definition, comment); the comma goes between the
            # definition and the comment so the rendered DDL stays valid-looking
            # (a trailing comma must not land inside a `--` comment).
            entries: list[tuple[str, str]] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(f"PRAGMA table_info({_q(t)})"):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                # Annotate non-PK columns with real example values; PKs are IDs,
                # not useful as literals.
                comment = "" if pk else _example_values(conn, t, name)
                entries.append((line, comment))
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                ref_table, from_col, to_col = fk[2], fk[3], fk[4]
                # SQLite reports `to` as NULL when the FK references the parent's
                # primary key implicitly; render without the column in that case.
                ref = f"{_q(ref_table)}({_q(to_col)})" if to_col is not None else _q(ref_table)
                entries.append((f"  FOREIGN KEY ({_q(from_col)}) REFERENCES {ref}", ""))
            rendered = []
            for i, (defn, comment) in enumerate(entries):
                sep = "," if i < len(entries) - 1 else ""
                rendered.append(f"{defn}{sep}  {comment}".rstrip() if comment else f"{defn}{sep}")
            parts.append("\n".join(rendered))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
