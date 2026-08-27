"""
install_mod.py — copy the GhostRace mod into the Balatro Mods folder.

    python -m ghost.install_mod [--uninstall]

The repo copy (ghost/mod/GhostRace/) is the source of truth; re-run after any change.
Balatro must be restarted to pick up mod changes (smods loads mods at boot).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent / "mod" / "GhostRace"


def mods_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise SystemExit("APPDATA is not set")
    d = Path(appdata) / "Balatro" / "Mods"
    if not d.is_dir():
        raise SystemExit(f"Balatro Mods folder not found at {d}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ghost.install_mod", description=__doc__)
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args(argv)

    dest = mods_dir() / "GhostRace"
    if args.uninstall:
        if dest.exists():
            shutil.rmtree(dest)
            print(f"removed {dest}")
        else:
            print(f"nothing at {dest}")
        return 0

    shutil.copytree(SRC, dest, dirs_exist_ok=True)
    print(f"installed {SRC} -> {dest}")
    print("Restart Balatro for the mod (re)load to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
